import torch
import torch.nn as nn
import argparse
from pathlib import Path

from robustbench.data import load_cifar10c, PREPROCESSINGS
from robustbench.loaders import CustomImageFolder


class FeatureCollector:
    def __init__(self, cls_only=True):
        self.features = []
        self.cls_only = cls_only

    def hook_fn(self, module, input, output):
        if self.cls_only:
            self.features.append(output[:, 0:1].detach().cpu())
        else:
            self.features.append(output.detach().cpu())
        return output

    def compute_stats(self):
        all_f = torch.cat(self.features, dim=0)
        mu = all_f.mean(dim=(0, 1), keepdim=True)
        sigma = all_f.std(dim=(0, 1), keepdim=True)
        return mu, sigma

    def reset(self):
        self.features = []


class EnvironmentMatrixSwapper:
    def __init__(self, cls_only=True):
        self.mode = 'passthrough'
        self.mu_src = None
        self.sigma_src = None
        self.cls_only = cls_only

    def set_target(self, mu, sigma):
        self.mu_src = mu
        self.sigma_src = sigma

    def hook_fn(self, module, input, output):
        if self.mode == 'passthrough':
            return output

        if self.mode == 'swap':
            assert self.mu_src is not None, "Target stats not set"
            if self.cls_only:
                f = output[:, 0:1]
                current_mu = f.mean(dim=(0, 1), keepdim=True)
                current_sigma = f.std(dim=(0, 1), keepdim=True)
                normalized = (f - current_mu) / (current_sigma + 1e-6)
                f_swapped = normalized * self.sigma_src + self.mu_src
                output = output.clone()
                output[:, 0:1] = f_swapped
                return output
            else:
                current_mu = output.mean(dim=(0, 1), keepdim=True)
                current_sigma = output.std(dim=(0, 1), keepdim=True)
                normalized = (output - current_mu) / (current_sigma + 1e-6)
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
            model = timm.create_model("vit_base_patch16_224", pretrained=args.pretrained)
        elif args.dataset == 'cifar10':
            model = timm.create_model("vit_base_patch16_384", pretrained=args.pretrained, num_classes=10)
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
    else:
        raise AttributeError("Cannot find transformer blocks in model")


def load_paired_imagenetc(n_examples, severity, data_dir, corruption_a, corruption_b):
    transform = PREPROCESSINGS['Res256Crop224']

    path_a = Path(data_dir) / corruption_a / str(severity)
    path_b = Path(data_dir) / corruption_b / str(severity)

    dataset_a = CustomImageFolder(str(path_a), transform)
    dataset_b = CustomImageFolder(str(path_b), transform)

    n = min(n_examples, len(dataset_a))
    loader_a = torch.utils.data.DataLoader(dataset_a, batch_size=n, shuffle=False)
    loader_b = torch.utils.data.DataLoader(dataset_b, batch_size=n, shuffle=False)

    x_a, y_a, paths_a = next(iter(loader_a))
    x_b, y_b, paths_b = next(iter(loader_b))

    assert torch.equal(y_a[:n], y_b[:n]), \
        f"Label mismatch: images not paired between '{corruption_a}' and '{corruption_b}'"

    rel_a = [str(Path(p).relative_to(path_a)) for p in paths_a[:n]]
    rel_b = [str(Path(p).relative_to(path_b)) for p in paths_b[:n]]
    assert rel_a == rel_b, \
        f"Filename mismatch: images not paired between '{corruption_a}' and '{corruption_b}'"

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_a = (x_a - mean) / std
    x_b = (x_b - mean) / std

    print(f"[*] Paired loading verified: {n} images, labels + filenames match")
    return x_a[:n], y_a[:n], x_b[:n], y_b[:n]


def load_paired_cifar10c(n_examples, severity, data_dir, corruption_a, corruption_b, resize=384):
    x_a, y_a = load_cifar10c(n_examples, severity, data_dir,
                             corruptions=[corruption_a])
    x_b, y_b = load_cifar10c(n_examples, severity, data_dir,
                             corruptions=[corruption_b])

    assert torch.equal(y_a, y_b), \
        "Label mismatch: CIFAR-10-C images not paired"

    x_a = torch.nn.functional.interpolate(x_a, size=(resize, resize), mode='bilinear', align_corners=False)
    x_b = torch.nn.functional.interpolate(x_b, size=(resize, resize), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    x_a = (x_a - mean) / std
    x_b = (x_b - mean) / std
    return x_a, y_a, x_b, y_b


def load_data(args):
    if args.dataset == 'imagenet':
        x_a, y_a, x_b, y_b = load_paired_imagenetc(
            args.n_examples, args.severity, args.data_dir,
            args.corruption_a, args.corruption_b)
    elif args.dataset == 'cifar10':
        resize = 224 if args.backend == 'torchvision' else 384
        x_a, y_a, x_b, y_b = load_paired_cifar10c(
            args.n_examples, args.severity, args.data_dir,
            args.corruption_a, args.corruption_b, resize=resize)
    n = min(x_a.shape[0], args.n_examples)
    x_a, y_a = x_a[:n], y_a[:n]
    x_b, y_b = x_b[:n], y_b[:n]
    return x_a, y_a, x_b, y_b


@torch.no_grad()
def batched_forward(model, x, batch_size, device):
    outputs = []
    for i in range(0, x.shape[0], batch_size):
        end = min(i + batch_size, x.shape[0])
        x_batch = x[i:end].to(device)
        outputs.append(model(x_batch))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def extract_domain_stats(model, x, layer, batch_size, device, cls_only):
    collector = FeatureCollector(cls_only=cls_only)
    handle = layer.register_forward_hook(collector.hook_fn)
    batched_forward(model, x, batch_size, device)
    mu, sigma = collector.compute_stats()
    handle.remove()
    n_tokens_info = 'CLS-only' if cls_only else 'all tokens'
    print(f"      Accumulated features from {x.shape[0]} images ({n_tokens_info})")
    return mu.to(device), sigma.to(device)


def run_swap_for_layer(model, layer, x_a, y_a, x_b, y_b, args, device):
    cls_only = not args.all_tokens

    print(f"\n  [Extracting domain statistics]")
    mu_A, sigma_A = extract_domain_stats(model, x_a, layer, args.batch_size, device, cls_only)
    mu_B, sigma_B = extract_domain_stats(model, x_b, layer, args.batch_size, device, cls_only)

    swapper = EnvironmentMatrixSwapper(cls_only=cls_only)
    handle = layer.register_forward_hook(swapper.hook_fn)

    swapper.mode = 'passthrough'
    out_a = batched_forward(model, x_a, args.batch_size, device)
    pred_a = out_a.argmax(dim=1).cpu()
    acc_a = (pred_a == y_a).float().mean().item()

    out_b = batched_forward(model, x_b, args.batch_size, device)
    pred_b = out_b.argmax(dim=1).cpu()
    acc_b = (pred_b == y_b).float().mean().item()

    swapper.mode = 'swap'
    swapper.set_target(mu_B, sigma_B)
    out_a2b = batched_forward(model, x_a, args.batch_size, device)
    pred_a2b = out_a2b.argmax(dim=1).cpu()
    acc_a2b = (pred_a2b == y_a).float().mean().item()
    cons_a2b = (pred_a == pred_a2b).float().mean().item()

    swapper.set_target(mu_A, sigma_A)
    out_b2a = batched_forward(model, x_b, args.batch_size, device)
    pred_b2a = out_b2a.argmax(dim=1).cpu()
    acc_b2a = (pred_b2a == y_b).float().mean().item()
    cons_b2a = (pred_b == pred_b2a).float().mean().item()

    handle.remove()
    return {
        'acc_a': acc_a, 'acc_b': acc_b,
        'acc_a2b': acc_a2b, 'acc_b2a': acc_b2a,
        'cons_a2b': cons_a2b, 'cons_b2a': cons_b2a,
    }


def run_swap_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    model = build_model(args, device)
    x_a, y_a, x_b, y_b = load_data(args)

    layers = args.scan_layers if args.scan_layers else [args.hook_layer]
    n_layers = len(model.encoder.layers) if hasattr(model, 'encoder') else len(model.blocks)
    layers = [l for l in layers if 0 <= l < n_layers]
    if not layers:
        raise ValueError(f"No valid layers in range [0, {n_layers - 1}]")

    header = f"\n{'='*75}"
    print(header)
    print(f" Environment Matrix Swap Validation")
    print(f" Dataset: {args.dataset} | Severity: {args.severity} | Backend: {args.backend}")
    print(f" Env A: {args.corruption_a}  |  Env B: {args.corruption_b}")
    print(f" Samples: {x_a.shape[0]} | Batch: {args.batch_size} |"
          f" Stats: {'CLS-only' if not args.all_tokens else 'all tokens'}")
    if args.scan_layers:
        print(f" Scanning layers: {layers} (of {n_layers} blocks)")
    else:
        print(f" Hook layer: block[{args.hook_layer}]")
    print(header)

    if args.clean_ref:
        from robustbench.data import load_clean_dataset
        from robustbench.model_zoo.enums import BenchmarkDataset
        ds_enum = BenchmarkDataset.imagenet if args.dataset == 'imagenet' else BenchmarkDataset.cifar_10
        prepr = 'Res256Crop224' if args.dataset == 'imagenet' else 'none'
        x_clean, y_clean = load_clean_dataset(ds_enum, args.n_examples, args.data_dir, prepr)
        x_clean = x_clean[:min(x_clean.shape[0], args.n_examples)]
        y_clean = y_clean[:min(y_clean.shape[0], args.n_examples)]
        if args.dataset == 'imagenet':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x_clean = (x_clean - mean) / std
        if args.dataset == 'cifar10':
            resize = 224 if args.backend == 'torchvision' else 384
            x_clean = torch.nn.functional.interpolate(x_clean, size=(resize, resize), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
            x_clean = (x_clean - mean) / std
        out_clean = batched_forward(model, x_clean, args.batch_size, device)
        pred_clean = out_clean.argmax(dim=1).cpu()
        acc_clean = (pred_clean == y_clean).float().mean().item()
        print(f"\n  [Clean reference] accuracy: {acc_clean:.2%}")

    results = {}
    for layer_idx in layers:
        layer = get_block(model, layer_idx)
        label = f"block[{layer_idx}]"
        print(f"\n{'─'*50}")
        print(f"  Layer: {label}")
        print(f"{'─'*50}")
        r = run_swap_for_layer(model, layer, x_a, y_a, x_b, y_b, args, device)
        results[label] = r

    print(f"\n{'='*75}")
    print(f" Results Summary")
    print(f"{'='*75}")
    print(f" {'Layer':<14} {'Acc_A':>8} {'Acc_B':>8} {'A→B':>8} {'B→A':>8}"
          f" {'Cons_A→B':>10} {'Cons_B→A':>10} {'ΔAcc_avg':>10}")
    print(f" {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}"
          f" {'─'*10} {'─'*10} {'─'*10}")
    for label, r in results.items():
        delta = ((r['acc_a'] - r['acc_a2b']) + (r['acc_b'] - r['acc_b2a'])) / 2
        print(f" {label:<14} {r['acc_a']:7.2%} {r['acc_b']:7.2%} {r['acc_a2b']:7.2%} {r['acc_b2a']:7.2%}"
              f" {r['cons_a2b']:9.2%} {r['cons_b2a']:9.2%} {delta:+9.2%}")
    print(f"{'='*75}")

    if args.clean_ref:
        print(f" Clean accuracy: {acc_clean:.2%}")
        for label, r in results.items():
            ratio_a2b = r['acc_a2b'] / acc_clean if acc_clean > 0 else 0
            ratio_b2a = r['acc_b2a'] / acc_clean if acc_clean > 0 else 0
            print(f" {label}: A→B retains {ratio_a2b:.1%} of clean acc,"
                  f" B→A retains {ratio_b2a:.1%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Environment Matrix Swap Validation')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--corruption_a', type=str, default='defocus_blur')
    parser.add_argument('--corruption_b', type=str, default='contrast')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hook_layer', type=int, default=1)
    parser.add_argument('--scan_layers', type=int, nargs='+', default=None,
                        help='Scan multiple layers, e.g. --scan_layers 0 1 3 5 7 9 11')
    parser.add_argument('--all_tokens', action='store_true',
                        help='Use all tokens for stats (default: CLS only, matching paper)')
    parser.add_argument('--clean_ref', action='store_true',
                        help='Include clean (uncorrupted) reference accuracy')
    parser.add_argument('--backend', type=str, default='torchvision',
                        choices=['torchvision', 'timm'],
                        help='Model backend (torchvision downloads from PyTorch CDN)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights (timm backend only)')
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained',
                        help='Skip pretrained weights')
    args = parser.parse_args()
    run_swap_experiment(args)
