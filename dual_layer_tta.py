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


class DualLayerConsistencyTTA:
    def __init__(self, model, extract_layer, swap_layer, beta=5.0,
                 alpha_blend=0.5, bank_size=20, lr=0.001, cls_only=True):
        self.model = model
        self.beta = beta
        self.alpha_blend = alpha_blend
        self.bank_size = bank_size

        self.collector = FeatureCollector(cls_only=cls_only)
        self.swapper = EnvironmentMatrixSwapper(cls_only=cls_only)

        self.extract_handle = extract_layer.register_forward_hook(
            self.collector.hook_fn)
        self.swap_handle = swap_layer.register_forward_hook(
            self.swapper.hook_fn)

        self.bank_mu = []
        self.bank_sigma = []

        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("No trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

    @torch.no_grad()
    def warmup_bank(self, x_warmup, batch_size):
        device = next(self.model.parameters()).device
        self.swapper.mode = 'passthrough'
        self.collector.reset()
        for i in range(0, min(x_warmup.shape[0], self.bank_size * batch_size),
                       batch_size):
            end = min(i + batch_size, x_warmup.shape[0])
            x_batch = x_warmup[i:end].to(device)
            _ = self.model(x_batch)
            mu, sigma = self.collector.compute_stats()
            self.collector.reset()
            self._add_to_bank(mu.to(device), sigma.to(device))

    def _add_to_bank(self, mu, sigma):
        if len(self.bank_mu) >= self.bank_size:
            self.bank_mu.pop(0)
            self.bank_sigma.pop(0)
        self.bank_mu.append(mu.detach().clone().cpu())
        self.bank_sigma.append(sigma.detach().clone().cpu())

    def adapt_step(self, x, base_loss_fn):
        device = x.device
        self.optimizer.zero_grad()

        self.swapper.mode = 'passthrough'
        self.collector.reset()
        logits_orig = self.model(x)

        current_mu, current_sigma = self.collector.compute_stats()
        current_mu = current_mu.to(device)
        current_sigma = current_sigma.to(device)
        self.collector.reset()

        loss_base = base_loss_fn(logits_orig)
        loss_cons = torch.tensor(0.0, device=device)

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

        self._add_to_bank(current_mu, current_sigma)
        self.swapper.mode = 'passthrough'

        return {
            'total': total_loss.item(),
            'base': loss_base.item(),
            'consistency': loss_cons.item() if isinstance(loss_cons, torch.Tensor)
            else loss_cons,
        }

    def remove_hooks(self):
        self.extract_handle.remove()
        self.swap_handle.remove()


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    model = build_model(args, device)
    x_a, y_a, x_b, y_b = load_data(args)

    for p in model.parameters():
        p.requires_grad = False

    head = None
    for _, m in model.named_modules():
        if hasattr(m, 'head'):
            head = m.head
            break
    if head is None:
        for _, m in model.named_modules():
            if hasattr(m, 'heads'):
                head = m.heads.head
                break
    for p in head.parameters():
        p.requires_grad = True

    extract_layer = get_block(model, args.extract_layer)
    swap_layer = get_block(model, args.swap_layer)

    n_layers = (len(model.encoder.layers) if hasattr(model, 'encoder')
                else len(model.blocks))

    print(f"\n{'='*70}")
    print(f" Dual-Layer Consistency TTA")
    print(f" Dataset: {args.dataset} | Severity: {args.severity}")
    print(f" Env A: {args.corruption_a}  |  Bank: {args.corruption_b}")
    print(f" Extract: block[{args.extract_layer}] (clean domain stats)")
    print(f" Inject:  block[{args.swap_layer}] (close to head, KL signal)")
    print(f" beta sweep | blend={args.blend} | bank={args.bank_size}")
    print(f"{'='*70}")

    for beta in args.betas:
        head_initial = {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(head_initial)

        tta = DualLayerConsistencyTTA(
            model=model, extract_layer=extract_layer,
            swap_layer=swap_layer, beta=beta,
            alpha_blend=args.blend, bank_size=args.bank_size,
            lr=args.lr, cls_only=not args.all_tokens,
        )

        tta.warmup_bank(x_b, args.batch_size)
        print(f"  Bank: {len(tta.bank_mu)} entries from env B")

        bs = args.batch_size
        n = x_a.shape[0]
        preds = []
        losses = []

        for i in range(0, n, bs):
            end = min(i + bs, n)
            x_batch = x_a[i:end].to(device)
            for _ in range(args.inner_steps):
                info = tta.adapt_step(
                    x_batch,
                    base_loss_fn=lambda logits: softmax_entropy(logits).mean(0),
                )
                losses.append(info)
            with torch.no_grad():
                out = model(x_batch)
            preds.append(out.argmax(dim=1).cpu())

        preds = torch.cat(preds)
        acc = (preds == y_a[:n]).float().mean().item()
        avg_base = sum(l['base'] for l in losses) / len(losses)
        avg_cons = sum(l['consistency'] for l in losses) / len(losses)

        print(f"  beta={beta:5.1f} | Acc: {acc:.2%} | "
              f"base_loss: {avg_base:.4f} | cons_loss: {avg_cons:.4f}")

        tta.remove_hooks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual-Layer Consistency TTA')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10'])
    parser.add_argument('--data_dir', type=str,
                        default='/root/data/picture/ImageNet-C')
    parser.add_argument('--corruption_a', type=str, default='defocus_blur')
    parser.add_argument('--corruption_b', type=str, default='contrast')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract_layer', type=int, default=1,
                        help='Shallow layer for clean domain stats extraction')
    parser.add_argument('--swap_layer', type=int, default=11,
                        help='Deep layer for swap (close to head)')
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.0, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument('--blend', type=float, default=0.5)
    parser.add_argument('--bank_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=3)
    parser.add_argument('--all_tokens', action='store_true')
    parser.add_argument('--backend', type=str, default='torchvision',
                        choices=['torchvision', 'timm'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false',
                        dest='pretrained')
    args = parser.parse_args()
    run_experiment(args)
