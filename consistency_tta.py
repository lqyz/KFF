import torch
import torch.nn.functional as F
import argparse

from validate_swap import (
    EnvironmentMatrixSwapper,
    FeatureCollector,
    build_model,
    load_data,
    get_block,
)


class ConsistencyTTA:
    def __init__(self, model, layer, beta=1.0, alpha_blend=0.5,
                 bank_size=20, lr=0.001, cls_only=True):
        self.model = model
        self.layer = layer
        self.beta = beta
        self.alpha_blend = alpha_blend
        self.bank_size = bank_size

        self.swapper = EnvironmentMatrixSwapper(cls_only=cls_only)
        self.collector = FeatureCollector(cls_only=cls_only)

        self.swapper_handle = layer.register_forward_hook(self.swapper.hook_fn)
        self.collector_handle = layer.register_forward_hook(self.collector.hook_fn)

        self.bank_mu = []
        self.bank_sigma = []

        trainable_params = []
        for p in model.parameters():
            if p.requires_grad:
                trainable_params.append(p)
        if not trainable_params:
            raise ValueError("No trainable parameters found. "
                             "Unfreeze some parameters or add prompts.")
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    @torch.no_grad()
    def warmup_bank(self, x_warmup, batch_size):
        device = next(self.model.parameters()).device
        self.swapper.mode = 'passthrough'
        self.collector.reset()
        for i in range(0, min(x_warmup.shape[0], self.bank_size * batch_size), batch_size):
            end = min(i + batch_size, x_warmup.shape[0])
            x_batch = x_warmup[i:end].to(device)
            _ = self.model(x_batch)
            mu, sigma = self.collector.compute_stats()
            self.collector.reset()
            self.update_bank(mu.to(device), sigma.to(device))

    def update_bank(self, mu, sigma):
        if len(self.bank_mu) >= self.bank_size:
            self.bank_mu.pop(0)
            self.bank_sigma.pop(0)
        self.bank_mu.append(mu.detach().clone().cpu())
        self.bank_sigma.append(sigma.detach().clone().cpu())

    def adapt_step(self, x, base_loss_fn):
        device = x.device
        self.optimizer.zero_grad()

        # Phase 1: original pass — extract stats AND get logits in one forward
        self.swapper.mode = 'passthrough'
        self.collector.reset()
        logits_orig = self.model(x)
        current_mu, current_sigma = self.collector.compute_stats()
        current_mu = current_mu.to(device)
        current_sigma = current_sigma.to(device)

        loss_base = base_loss_fn(logits_orig)
        self.collector.reset()
        loss_cons = torch.tensor(0.0, device=device)

        # Phase 2: augmented pass with blended historical domain
        if len(self.bank_mu) > 0:
            idx = torch.randint(0, len(self.bank_mu), (1,)).item()
            hist_mu = self.bank_mu[idx].to(device)
            hist_sigma = self.bank_sigma[idx].to(device)

            w = self.alpha_blend
            aug_mu = w * current_mu + (1 - w) * hist_mu
            aug_sigma = w * current_sigma + (1 - w) * hist_sigma

            self.swapper.mode = 'swap'
            self.swapper.set_target(aug_mu, aug_sigma)
            logits_aug = self.model(x)

            p_orig = F.softmax(logits_orig.detach(), dim=-1)
            log_p_aug = F.log_softmax(logits_aug, dim=-1)
            loss_cons = F.kl_div(log_p_aug, p_orig, reduction='batchmean')

        total_loss = loss_base + self.beta * loss_cons
        total_loss.backward()
        self.optimizer.step()

        self.update_bank(current_mu, current_sigma)
        self.swapper.mode = 'passthrough'

        return {
            'total': total_loss.item(),
            'base': loss_base.item(),
            'consistency': loss_cons.item() if isinstance(loss_cons, torch.Tensor) else loss_cons,
        }

    def remove_hooks(self):
        self.swapper_handle.remove()
        self.collector_handle.remove()


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def run_consistency_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    model = build_model(args, device)
    x_a, y_a, x_b, y_b = load_data(args)

    for p in model.parameters():
        p.requires_grad = False

    head = None
    for name, module in model.named_modules():
        if hasattr(module, 'head'):
            head = module.head
            break
    if head is None:
        for name, module in model.named_modules():
            if hasattr(module, 'heads'):
                head = module.heads.head
                break
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True
        print(f"[*] Unfroze classification head for adaptation")
    else:
        raise RuntimeError("Cannot find classification head to unfreeze")

    head_initial = {k: v.clone() for k, v in head.state_dict().items()}

    layer = get_block(model, args.hook_layer)

    for beta in args.betas:
        head.load_state_dict(head_initial)
        print(f"\n{'='*60}")
        print(f" beta = {beta} | layer = block[{args.hook_layer}] |"
              f" blend = {args.alpha_blend} | bank = {args.bank_size} |"
              f" steps = {args.inner_steps}")
        if args.cross_domain:
            print(f" cross-domain: bank warmed on {args.corruption_b}, "
                  f"adapting on {args.corruption_a}")
        print(f"{'='*60}")

        tta = ConsistencyTTA(
            model=model, layer=layer, beta=beta,
            alpha_blend=args.alpha_blend, bank_size=args.bank_size,
            lr=args.lr, cls_only=not args.all_tokens,
        )

        if args.cross_domain:
            tta.warmup_bank(x_b, args.batch_size)

        bs = args.batch_size
        n_samples = x_a.shape[0]

        preds_after = []
        losses_log = []

        for i in range(0, n_samples, bs):
            end = min(i + bs, n_samples)
            x_batch = x_a[i:end].to(device)

            for _ in range(args.inner_steps):
                info = tta.adapt_step(
                    x_batch,
                    base_loss_fn=lambda logits: softmax_entropy(logits).mean(0),
                )
                losses_log.append(info)

            with torch.no_grad():
                self_out = model(x_batch)
            preds_after.append(self_out.argmax(dim=1).cpu())

        preds = torch.cat(preds_after)
        acc = (preds == y_a[:n_samples]).float().mean().item()

        avg_base = sum(l['base'] for l in losses_log) / len(losses_log)
        avg_cons = sum(l['consistency'] for l in losses_log) / len(losses_log)
        avg_total = sum(l['total'] for l in losses_log) / len(losses_log)

        print(f"  Acc: {acc:.2%}")
        print(f"  Avg losses — base: {avg_base:.4f}  "
              f"consistency: {avg_cons:.4f}  total: {avg_total:.4f}")

        tta.remove_hooks()

        for p in model.parameters():
            if p.requires_grad:
                p.requires_grad = False
                p.grad = None
        for p in head.parameters():
            p.requires_grad = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consistency TTA Experiment')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--corruption_a', type=str, default='defocus_blur')
    parser.add_argument('--corruption_b', type=str, default='contrast')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hook_layer', type=int, default=5)
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument('--alpha_blend', type=float, default=0.5)
    parser.add_argument('--bank_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=1,
                        help='Adaptation steps per batch')
    parser.add_argument('--all_tokens', action='store_true')
    parser.add_argument('--backend', type=str, default='torchvision',
                        choices=['torchvision', 'timm'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    parser.add_argument('--cross_domain', action='store_true',
                        help='Warm bank with env B, adapt on env A (cross-domain)')
    args = parser.parse_args()
    run_consistency_experiment(args)
