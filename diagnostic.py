import torch
import timm
import argparse
import numpy as np
from pathlib import Path
from robustbench.data import load_imagenetc, PREPROCESSINGS
from robustbench.loaders import CustomImageFolder

def compute_tss(features):
    """Token Spatial Specificity:
    TSS = Var_over_tokens(Mean_over_channels(f)) / Mean_over_tokens(Var_over_channels(f))
    features: [B, N, D]"""
    mean_c = features.mean(dim=-1)      # [B, N] — per-token channel mean
    var_tok = mean_c.var(dim=1)          # [B]   — spatial variance of channel means
    var_c = features.var(dim=-1)         # [B, N] — per-token channel variance
    mean_tok = var_c.mean(dim=1)         # [B]   — mean of channel variances
    tss = (var_tok / (mean_tok + 1e-8)).mean().item()
    return tss


def collect_attention_entropy(model, x, layer_idx):
    """Attention Matrix Entropy for a specific layer."""
    attn_weights = []

    def hook(module, input, output):
        # timm stores attn weights differently; capture via QK computation
        pass

    if hasattr(model, 'blocks'):
        block = model.blocks[layer_idx]
    else:
        block = model.encoder.layers[layer_idx]

    # Register hook on qkv to capture attention
    # ViT attention: qkv = Linear(x) -> q, k, v = split(qkv)
    # attn = softmax(q @ k.T / sqrt(d)) — need to capture this
    # timm doesn't expose attn weights by default; use output_attentions in newer versions
    # For now, approximate AME via output feature entropy
    with torch.no_grad():
        out = block(x)
        # AME proxy: std of CLS token after attention (clean → more structure)
        cls_std = out[:, 0].std(dim=0).norm().item()
    return cls_std


def diagnostic_run(args):
    import torchvision.transforms as transforms
    from PIL import Image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Diagnostic: TSS curves for clean vs corrupted domains")

    model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
    model.eval()

    transform = PREPROCESSINGS['Res256Crop224']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Load data
    pairs = [
        ('clean', None),
        ('gaussian_noise', 'gaussian_noise'),
        ('defocus_blur', 'defocus_blur'),
        ('contrast', 'contrast'),
        ('brightness', 'brightness'),
        ('fog', 'fog'),
        ('snow', 'snow'),
    ]

    results = {}
    c_data = args.corruption_data_dir

    for label, corr in pairs:
        if corr:
            x_all, _ = load_imagenetc(min(args.n_examples, 200), args.severity,
                                      c_data, corruptions=[corr])
            x_all = (x_all - mean) / std
        else:  # clean
            path = Path(args.data_dir) / 'val' / 'images'
            imgs = sorted(Path(path).glob('*.JPEG'))[:min(args.n_examples, 200)]
            T = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor()])
            x_list = [T(Image.open(p).convert('RGB')) for p in imgs]
            x_all = torch.stack(x_list)
            x_all = (x_all - mean) / std

        bs = args.batch_size
        n = min(x_all.shape[0], args.n_examples)
        x_all = x_all[:n]

        n_blocks = len(model.blocks)
        tss_curve = []
        cls_std_curve = []

        handles = []

        def make_tss_hook(idx):
            def hook(module, input, output):
                tss_curve[idx] += compute_tss(output) * 1
                cls_std_curve[idx] += output[:, 0].std(dim=0).norm().item()
                return output
            return hook

        for i in range(n_blocks):
            tss_curve.append(0.0)
            cls_std_curve.append(0.0)
            handles.append(model.blocks[i].register_forward_hook(
                make_tss_hook(i)))

        n_batches = 0
        for i in range(0, n, bs):
            end = min(i + bs, n)
            _ = model(x_all[i:end].to(device))
            n_batches += 1

        for i in range(n_blocks):
            tss_curve[i] /= n_batches
            cls_std_curve[i] /= n_batches

        for h in handles:
            h.remove()

        results[label] = {'tss': tss_curve, 'cls_std': cls_std_curve}

    # Print results
    probe = [0, 2, 4, 6, 8, 10]
    print(f"\n{'='*80}")
    print(f"{'Domain':<20}", end='')
    for p in probe:
        print(f"{'b'+str(p):>10}", end='')
    print(f"{' avg':>10}")
    print('-' * 80)

    for label, r in results.items():
        print(f"{label:<20}", end='')
        vals = [r['tss'][p] for p in probe]
        for v in vals:
            print(f"{v:10.4f}", end='')
        print(f"{sum(vals)/len(vals):10.4f}")

    print(f"\n{'='*80}")
    print(f" CLS std (proxy for attention structure)")
    print(f"{'Domain':<20}", end='')
    for p in probe:
        print(f"{'b'+str(p):>10}", end='')
    print('-' * 80)
    for label, r in results.items():
        print(f"{label:<20}", end='')
        vals = [r['cls_std'][p] for p in probe]
        for v in vals:
            print(f"{v:10.2f}", end='')
        print()

    # Key metric: slope from block 2 to 8 (semantic emergence region)
    print(f"\n{'='*80}")
    print(f" TSS slope (b2→b8): semantic emergence gradient")
    print(f"{'Domain':<20} {'slope':>10} {'b8/b2 ratio':>12}")
    print('-' * 80)
    for label, r in results.items():
        slope = r['tss'][8] - r['tss'][2]
        ratio = r['tss'][8] / (r['tss'][2] + 1e-8)
        print(f"{label:<20} {slope:10.4f} {ratio:12.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSS/AME Diagnostic')
    parser.add_argument('--data_dir', type=str,
                        default='/root/data/imagenet',
                        help='ImageNet root (for clean val images)')
    parser.add_argument('--corruption_data_dir', type=str,
                        default='/root/data/picture/ImageNet-C',
                        help='ImageNet-C root (for corrupted images)')
    parser.add_argument('--n_examples', type=int, default=200)
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    diagnostic_run(args)
