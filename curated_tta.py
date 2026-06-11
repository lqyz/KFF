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


class CuratedMemoryBank:
    def __init__(self, max_size=20, thr_dist=25.0, ema_alpha=0.9,
                 thr_affinity=0.3, prune_interval=10):
        self.max_size = max_size
        self.thr_dist = thr_dist
        self.ema_alpha = ema_alpha
        self.thr_affinity = thr_affinity
        self.prune_interval = prune_interval

        self.entries = []   # list of (mu, sigma, affinity, count)
        self.step = 0

    def _find_nearest(self, mu):
        if not self.entries:
            return -1, float('inf')
        dists = [(torch.norm(mu - e[0], p=2).item(), i)
                 for i, e in enumerate(self.entries)]
        dists.sort(key=lambda x: x[0])
        return dists[0][1], dists[0][0]

    def update(self, mu, sigma):
        mu_cpu = mu.detach().cpu().clone()
        sigma_cpu = sigma.detach().cpu().clone()

        nearest_idx, min_dist = self._find_nearest(mu_cpu)

        if nearest_idx < 0 or min_dist > self.thr_dist:
            if len(self.entries) >= self.max_size:
                self.entries.sort(key=lambda e: e[2])  # by affinity
                self.entries.pop(0)
            self.entries.append([mu_cpu, sigma_cpu, 0.5, 1])
        else:
            a = self.ema_alpha
            e = self.entries[nearest_idx]
            e[0] = a * e[0] + (1 - a) * mu_cpu
            e[1] = a * e[1] + (1 - a) * sigma_cpu
            e[3] += 1

    def record_affinity(self, idx, affinity):
        if 0 <= idx < len(self.entries):
            a = 0.9
            self.entries[idx][2] = a * self.entries[idx][2] + (1 - a) * affinity

        self.step += 1
        if self.prune_interval > 0 and self.step % self.prune_interval == 0:
            self.prune()

    def sample_weighted(self):
        if not self.entries:
            return None, None, -1
        weights = torch.tensor([max(e[2], 1e-6) for e in self.entries])
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, 1).item()
        return self.entries[idx][0], self.entries[idx][1], idx

    def prune(self):
        before = len(self.entries)
        self.entries = [e for e in self.entries if e[2] >= self.thr_affinity]
        after = len(self.entries)
        if before != after:
            print(f"      [Bank prune] {before} -> {after} entries "
                  f"(thr={self.thr_affinity})")

    def __len__(self):
        return len(self.entries)

    def stats(self):
        if not self.entries:
            return "empty"
        affs = [e[2] for e in self.entries]
        return (f"{len(self.entries)} entries, "
                f"affinity: {min(affs):.3f}~{max(affs):.3f}")


class CuratedConsistencyTTA:
    def __init__(self, model, layer, bank, beta=5.0, alpha_blend=0.5,
                 lr=0.001, cls_only=True):
        self.model = model
        self.layer = layer
        self.bank = bank
        self.beta = beta
        self.alpha_blend = alpha_blend

        self.swapper = EnvironmentMatrixSwapper(cls_only=cls_only)
        self.collector = FeatureCollector(cls_only=cls_only)

        self.swapper_handle = layer.register_forward_hook(
            self.swapper.hook_fn)
        self.collector_handle = layer.register_forward_hook(
            self.collector.hook_fn)

        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr)

    def warmup_bank(self, x_warmup, batch_size):
        device = next(self.model.parameters()).device
        self.swapper.mode = 'passthrough'
        for i in range(0, min(x_warmup.shape[0],
                              self.bank.max_size * batch_size), batch_size):
            end = min(i + batch_size, x_warmup.shape[0])
            x_batch = x_warmup[i:end].to(device)
            self.collector.reset()
            _ = self.model(x_batch)
            mu, sigma = self.collector.compute_stats()
            self.bank.update(mu.to(device), sigma.to(device))

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
        bank_idx = -1

        hist_mu, hist_sigma, bank_idx = self.bank.sample_weighted()
        if hist_mu is not None:
            w = self.alpha_blend
            aug_mu = w * current_mu + (1 - w) * hist_mu.to(device)
            aug_sigma = w * current_sigma + (1 - w) * hist_sigma.to(device)

            self.swapper.mode = 'swap'
            self.swapper.set_target(aug_mu, aug_sigma)
            logits_aug = self.model(x)

            p_orig = F.softmax(logits_orig.detach(), dim=-1)
            log_p_aug = F.log_softmax(logits_aug, dim=-1)
            loss_cons = F.kl_div(log_p_aug, p_orig, reduction='batchmean')

            h_orig = -(p_orig * torch.log(p_orig + 1e-6)).sum(1)
            p_aug = F.softmax(logits_aug, dim=-1)
            h_aug = -(p_aug * torch.log(p_aug + 1e-6)).sum(1)
            delta_h = (h_aug - h_orig).mean().item()
            affinity = float(torch.exp(-max(delta_h, 0) / 1.0))
            self.bank.record_affinity(bank_idx, affinity)

        total_loss = loss_base + self.beta * loss_cons
        total_loss.backward()
        self.optimizer.step()

        self.bank.update(current_mu, current_sigma)
        self.swapper.mode = 'passthrough'

        return {
            'total': total_loss.item(),
            'base': loss_base.item(),
            'consistency': loss_cons.item() if isinstance(
                loss_cons, torch.Tensor) else loss_cons,
        }

    def remove_hooks(self):
        self.swapper_handle.remove()
        self.collector_handle.remove()


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def run_curated_experiment(args):
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

    layer = get_block(model, args.hook_layer)
    head_initial = {k: v.clone() for k, v in head.state_dict().items()}

    print(f"\n{'='*70}")
    print(f" Curated Memory Bank Consistency TTA")
    print(f" Dataset: {args.dataset} | Severity: {args.severity}")
    print(f" Env A: {args.corruption_a}  |  Bank: {args.corruption_b}")
    print(f" Hook: block[{args.hook_layer}] | blend={args.blend} | "
          f"prune_thr={args.prune_thr}")
    print(f"{'='*70}")

    for beta in args.betas:
        head.load_state_dict(head_initial)

        bank = CuratedMemoryBank(
            max_size=args.bank_size, thr_dist=args.thr_d,
            ema_alpha=0.9, thr_affinity=args.prune_thr,
            prune_interval=args.prune_interval)

        tta = CuratedConsistencyTTA(
            model=model, layer=layer, bank=bank, beta=beta,
            alpha_blend=args.blend, lr=args.lr,
            cls_only=not args.all_tokens,
        )

        tta.warmup_bank(x_b, args.batch_size)
        print(f"  [Bank after warmup] {bank.stats()}")

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
        cons_signal = avg_cons > 0.001

        print(f"  beta={beta:5.1f} | Acc: {acc:.2%} | "
              f"base: {avg_base:.4f} | cons: {avg_cons:.4f} | "
              f"signal: {cons_signal} | {bank.stats()}")

        tta.remove_hooks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Curated Memory Bank TTA')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10'])
    parser.add_argument('--data_dir', type=str,
                        default='/root/data/picture/ImageNet-C')
    parser.add_argument('--corruption_a', type=str, default='defocus_blur')
    parser.add_argument('--corruption_b', type=str, default='contrast')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hook_layer', type=int, default=11)
    parser.add_argument('--betas', type=float, nargs='+',
                        default=[0.0, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument('--blend', type=float, default=0.5)
    parser.add_argument('--bank_size', type=int, default=20)
    parser.add_argument('--thr_d', type=float, default=25.0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--inner_steps', type=int, default=3)
    parser.add_argument('--prune_thr', type=float, default=0.3)
    parser.add_argument('--prune_interval', type=int, default=10)
    parser.add_argument('--all_tokens', action='store_true')
    parser.add_argument('--backend', type=str, default='torchvision',
                        choices=['torchvision', 'timm'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false',
                        dest='pretrained')
    args = parser.parse_args()
    run_curated_experiment(args)
