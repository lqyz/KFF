import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path

from robustbench.data import load_cifar10c, PREPROCESSINGS
from robustbench.loaders import CustomImageFolder


class EnvironmentMemoryBank:
    def __init__(self, max_size=20, threshold=25.0, ema_alpha=0.9):
        self.bank = []
        self.max_size = max_size
        self.threshold = threshold
        self.ema_alpha = ema_alpha

    def update(self, mu, sigma):
        mu_det = mu.detach().cpu().clone()
        sigma_det = sigma.detach().cpu().clone()

        if len(self.bank) == 0:
            self.bank.append((mu_det, sigma_det))
            return

        distances = [torch.norm(mu_det - b_mu, p=2).item()
                     for b_mu, _ in self.bank]
        min_dist = min(distances)
        min_idx = distances.index(min_dist)

        if min_dist > self.threshold:
            while len(self.bank) >= self.max_size:
                self.bank.pop(0)
            self.bank.append((mu_det, sigma_det))
        else:
            a = self.ema_alpha
            self.bank[min_idx] = (
                a * self.bank[min_idx][0] + (1 - a) * mu_det,
                a * self.bank[min_idx][1] + (1 - a) * sigma_det,
            )

    def sample_random(self):
        if len(self.bank) == 0:
            return None, None
        idx = torch.randint(0, len(self.bank), (1,)).item()
        return self.bank[idx]

    def __len__(self):
        return len(self.bank)


class EnvironmentMatrixSwapper:
    def __init__(self, cls_only=True):
        self.mode = 'passthrough'
        self.mu_src = None
        self.sigma_src = None
        self.current_mu = None
        self.current_sigma = None
        self.cls_only = cls_only

    def set_target(self, mu, sigma):
        self.mu_src = mu
        self.sigma_src = sigma

    def hook_fn(self, module, input, output):
        f_stat = output[:, 0:1] if self.cls_only else output
        self.current_mu = f_stat.mean(dim=(0, 1), keepdim=True).detach()
        self.current_sigma = f_stat.std(dim=(0, 1), keepdim=True).detach()

        if self.mode == 'passthrough':
            return output

        if self.mode == 'swap':
            assert self.mu_src is not None, "Target stats not set"
            if self.cls_only:
                f = output[:, 0:1]
                cur_mu = f.mean(dim=(0, 1), keepdim=True)
                cur_sigma = f.std(dim=(0, 1), keepdim=True)
                normalized = (f - cur_mu) / (cur_sigma + 1e-6)
                f_swapped = normalized * self.sigma_src + self.mu_src
                output = output.clone()
                output[:, 0:1] = f_swapped
                return output
            else:
                cur_mu = output.mean(dim=(0, 1), keepdim=True)
                cur_sigma = output.std(dim=(0, 1), keepdim=True)
                normalized = (output - cur_mu) / (cur_sigma + 1e-6)
                return normalized * self.sigma_src + self.mu_src
        return output


def build_model(args, device):
    if args.backend == 'torchvision':
        import torchvision
        if args.dataset == 'imagenet':
            model = torchvision.models.vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        elif args.dataset == 'cifar10':
            model = torchvision.models.VisionTransformer(
                image_size=224, patch_size=16, num_layers=12, num_heads=12,
                hidden_dim=768, mlp_dim=3072, num_classes=10)
            state_dict = torchvision.models.vit_b_16(
                weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1).state_dict()
            del state_dict['heads.head.weight'], state_dict['heads.head.bias']
            model.load_state_dict(state_dict, strict=False)
    else:
        import timm
        if args.dataset == 'imagenet':
            model = timm.create_model("vit_base_patch16_224",
                                      pretrained=args.pretrained)
        elif args.dataset == 'cifar10':
            model = timm.create_model("vit_base_patch16_384",
                                      pretrained=args.pretrained, num_classes=10)

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_block(model, idx):
    if hasattr(model, 'blocks'):
        return model.blocks[idx]
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model.encoder.layers[idx]
    raise AttributeError("Cannot find transformer blocks in model")


def load_paired_imagenetc(n_examples, severity, data_dir, corruption_a,
                          corruption_b, batch_size=32):
    transform = PREPROCESSINGS['Res256Crop224']
    path_a = Path(data_dir) / corruption_a / str(severity)
    path_b = Path(data_dir) / corruption_b / str(severity)

    dataset_a = CustomImageFolder(str(path_a), transform)
    dataset_b = CustomImageFolder(str(path_b), transform)

    n = min(n_examples, len(dataset_a))
    loader_a = torch.utils.data.DataLoader(dataset_a, batch_size=batch_size,
                                           shuffle=False)
    loader_b = torch.utils.data.DataLoader(dataset_b, batch_size=batch_size,
                                           shuffle=False)

    x_a_list, y_a_list, x_b_list, y_b_list = [], [], [], []
    curr_count = 0
    for (xa, ya, _), (xb, yb, _) in zip(loader_a, loader_b):
        x_a_list.append(xa)
        y_a_list.append(ya)
        x_b_list.append(xb)
        y_b_list.append(yb)
        curr_count += xa.shape[0]
        if curr_count >= n:
            break

    x_a = torch.cat(x_a_list, dim=0)[:n]
    y_a = torch.cat(y_a_list, dim=0)[:n]
    x_b = torch.cat(x_b_list, dim=0)[:n]
    y_b = torch.cat(y_b_list, dim=0)[:n]

    assert torch.equal(y_a, y_b), "Label mismatch between domains"

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_a = (x_a - mean) / std
    x_b = (x_b - mean) / std

    print(f"[*] Paired ImageNet-C: {n} images matched.")
    return x_a, y_a, x_b, y_b


def load_paired_cifar10c(n_examples, severity, data_dir, corruption_a,
                         corruption_b, resize=384):
    x_a, y_a = load_cifar10c(n_examples, severity, data_dir,
                             corruptions=[corruption_a])
    x_b, y_b = load_cifar10c(n_examples, severity, data_dir,
                             corruptions=[corruption_b])
    assert torch.equal(y_a, y_b), "Label mismatch: CIFAR-10-C not paired"

    x_a = F.interpolate(x_a, size=(resize, resize),
                         mode='bilinear', align_corners=False)
    x_b = F.interpolate(x_b, size=(resize, resize),
                         mode='bilinear', align_corners=False)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    x_a = (x_a - mean) / std
    x_b = (x_b - mean) / std
    return x_a, y_a, x_b, y_b


def load_data(args):
    if args.dataset == 'imagenet':
        return load_paired_imagenetc(args.n_examples, args.severity,
                                     args.data_dir, args.corruption_a,
                                     args.corruption_b, args.batch_size)
    elif args.dataset == 'cifar10':
        resize = 224 if args.backend == 'torchvision' else 384
        return load_paired_cifar10c(args.n_examples, args.severity,
                                    args.data_dir, args.corruption_a,
                                    args.corruption_b, resize=resize)


@torch.no_grad()
def warmup_bank(model, layer, x_warmup, batch_size, device, cls_only,
                memory_bank, max_warmup_batches=None):
    swapper = EnvironmentMatrixSwapper(cls_only=cls_only)
    handle = layer.register_forward_hook(swapper.hook_fn)

    n = x_warmup.shape[0]
    b = 0
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        model(x_warmup[i:end].to(device))
        if swapper.current_mu is not None:
            memory_bank.update(swapper.current_mu, swapper.current_sigma)
        b += 1
        if max_warmup_batches is not None and b >= max_warmup_batches:
            break

    handle.remove()
    print(f"    Bank: {len(memory_bank)} environments registered "
          f"({b} batches)")


def run_tta_adaptation(model, layer, x_data, y_data, args, device, memory_bank):
    cls_only = not args.all_tokens
    swapper = EnvironmentMatrixSwapper(cls_only=cls_only)
    handle = layer.register_forward_hook(swapper.hook_fn)

    params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            params.append(m.weight)
            params.append(m.bias)
    if not params:
        raise RuntimeError("No LayerNorm layers found to adapt")
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    correct_before = 0
    correct_after = 0
    total = x_data.shape[0]
    affinities = []

    for i in range(0, total, args.batch_size):
        end = min(i + args.batch_size, total)
        x_batch = x_data[i:end].to(device)
        y_batch = y_data[i:end].to(device)

        swapper.mode = 'passthrough'
        with torch.no_grad():
            logits_pre = model(x_batch)
            correct_before += (logits_pre.argmax(1) == y_batch).sum().item()

        logits_orig = model(x_batch)
        p_orig = F.softmax(logits_orig, dim=-1)
        h_orig = -(p_orig * torch.log(p_orig + 1e-6)).sum(dim=-1)
        loss_entropy = h_orig.mean()

        loss_cons = torch.tensor(0.0, device=device)
        w_affinity_mean = 0.0

        hist_mu, hist_sigma = memory_bank.sample_random()
        if hist_mu is not None:
            swapper.mode = 'swap'
            swapper.set_target(hist_mu.to(device), hist_sigma.to(device))

            logits_swap = model(x_batch)
            p_swap = F.softmax(logits_swap, dim=-1)
            h_swap = -(p_swap * torch.log(p_swap + 1e-6)).sum(dim=-1)

            delta_h = h_swap - h_orig.detach()
            w_affinity = torch.exp(-torch.clamp(delta_h, min=0) / args.tau)
            w_affinity_mean = w_affinity.mean().item()
            affinities.append(w_affinity_mean)

            kl_loss = F.kl_div(
                F.log_softmax(logits_swap, dim=-1),
                p_orig.detach(),
                reduction='none',
            ).sum(dim=-1)
            loss_cons = (w_affinity * kl_loss).mean()

        total_loss = loss_entropy + args.beta * loss_cons

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            logits_after = model(x_batch)
            correct_after += (logits_after.argmax(1) == y_batch).sum().item()

        if swapper.current_mu is not None:
            memory_bank.update(swapper.current_mu, swapper.current_sigma)

        swapper.mode = 'passthrough'

    handle.remove()
    avg_affinity = sum(affinities) / len(affinities) if affinities else 0.0

    return {
        'acc_before': correct_before / total,
        'acc_after': correct_after / total,
        'avg_affinity': avg_affinity,
    }


def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    x_a, y_a, x_b, y_b = load_data(args)

    layers = args.scan_layers if args.scan_layers else [args.hook_layer]

    preload_model = build_model(args, device)
    cls_only = not args.all_tokens
    bank_layer = get_block(preload_model, layers[0])
    memory_bank = EnvironmentMemoryBank(
        max_size=args.bank_size, threshold=args.thr_d, ema_alpha=args.ema_alpha)

    print("[*] Warming bank with reference Env B statistics...")
    warmup_bank(preload_model, bank_layer, x_b, args.batch_size, device,
                cls_only, memory_bank, max_warmup_batches=args.warmup_batches)
    del preload_model

    header = f"\n{'='*75}"
    print(header)
    print(f" Affinity-Guided Consistency Regularization TTA")
    print(f" Dataset: {args.dataset} | Severity: {args.severity}")
    print(f" Env A: {args.corruption_a}  |  Bank: {args.corruption_b}")
    print(f" beta={args.beta} | tau={args.tau} | lr={args.lr} | "
          f"bank_size={args.bank_size}")
    print(header)

    for layer_idx in layers:
        model = build_model(args, device)
        layer = get_block(model, layer_idx)

        print(f"\n>> Block [{layer_idx}] ...")
        res = run_tta_adaptation(model, layer, x_a, y_a, args, device,
                                 memory_bank)

        print(f"   Before: {res['acc_before']:.2%}  "
              f"After: {res['acc_after']:.2%}  "
              f"Delta: {res['acc_after'] - res['acc_before']:+.2%}  "
              f"Affinity: {res['avg_affinity']:.3f}")
    print(f"{'='*75}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Affinity-Guided Consistency TTA')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10'])
    parser.add_argument('--data_dir', type=str,
                        default='/root/data/picture/ImageNet-C')
    parser.add_argument('--corruption_a', type=str, default='defocus_blur')
    parser.add_argument('--corruption_b', type=str, default='contrast')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hook_layer', type=int, default=11)
    parser.add_argument('--scan_layers', type=int, nargs='+', default=None)
    parser.add_argument('--all_tokens', action='store_true')
    parser.add_argument('--backend', type=str, default='torchvision',
                        choices=['torchvision', 'timm'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained')

    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bank_size', type=int, default=20)
    parser.add_argument('--thr_d', type=float, default=25.0)
    parser.add_argument('--ema_alpha', type=float, default=0.9)
    parser.add_argument('--warmup_batches', type=int, default=None)

    args = parser.parse_args()
    run_experiment(args)
