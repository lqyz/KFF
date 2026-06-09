import torch
import timm
import argparse
from pathlib import Path

from robustbench.data import load_cifar10c, PREPROCESSINGS
from robustbench.loaders import CustomImageFolder


class EnvironmentMatrixSwapper:
    def __init__(self):
        self.mode = 'passthrough'
        self.mu_A = None
        self.sigma_A = None
        self.mu_B = None
        self.sigma_B = None

    def hook_fn(self, module, input, output):
        if self.mode == 'extract_A':
            self.mu_A = output.mean(dim=(0, 1), keepdim=True)
            self.sigma_A = output.std(dim=(0, 1), keepdim=True)
            return output
        elif self.mode == 'extract_B':
            self.mu_B = output.mean(dim=(0, 1), keepdim=True)
            self.sigma_B = output.std(dim=(0, 1), keepdim=True)
            return output
        elif self.mode == 'swap_A_to_B':
            current_mu = output.mean(dim=(0, 1), keepdim=True)
            current_sigma = output.std(dim=(0, 1), keepdim=True)
            normalized_content = (output - current_mu) / (current_sigma + 1e-6)
            swapped_output = normalized_content * self.sigma_B + self.mu_B
            return swapped_output
        return output


def build_model(args, device):
    if args.dataset == 'imagenet':
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
    elif args.dataset == 'cifar10':
        model = timm.create_model("vit_base_patch16_384", pretrained=True, num_classes=10)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


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
        f"Label mismatch: images are not paired between '{corruption_a}' and '{corruption_b}'"

    rel_a = [str(Path(p).relative_to(path_a)) for p in paths_a[:n]]
    rel_b = [str(Path(p).relative_to(path_b)) for p in paths_b[:n]]
    assert rel_a == rel_b, \
        f"Filename mismatch: images are not paired between '{corruption_a}' and '{corruption_b}'"

    print(f"[*] Paired loading verified: {n} images, labels + filenames match")
    return x_a[:n], y_a[:n], x_b[:n], y_b[:n]


def load_paired_cifar10c(n_examples, severity, data_dir, corruption_a, corruption_b):
    x_a, y_a = load_cifar10c(n_examples, severity, data_dir,
                             corruptions=[corruption_a])
    x_b, y_b = load_cifar10c(n_examples, severity, data_dir,
                             corruptions=[corruption_b])

    assert torch.equal(y_a, y_b), \
        f"Label mismatch: CIFAR-10-C images are not paired"

    x_a = torch.nn.functional.interpolate(x_a, size=(384, 384), mode='bilinear', align_corners=False)
    x_b = torch.nn.functional.interpolate(x_b, size=(384, 384), mode='bilinear', align_corners=False)
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
        x_a, y_a, x_b, y_b = load_paired_cifar10c(
            args.n_examples, args.severity, args.data_dir,
            args.corruption_a, args.corruption_b)
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


def run_swap_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    model = build_model(args, device)
    x_a, y_a, x_b, y_b = load_data(args)

    swapper = EnvironmentMatrixSwapper()
    hook_layer = model.blocks[args.hook_layer]
    hook_handle = hook_layer.register_forward_hook(swapper.hook_fn)

    print(f"\n{'='*60}")
    print(f" Environment Matrix Swap Validation")
    print(f" Dataset: {args.dataset} | Severity: {args.severity}")
    print(f" Env A (corruption): {args.corruption_a}")
    print(f" Env B (corruption): {args.corruption_b}")
    print(f" Hook: blocks[{args.hook_layer}] | Samples: {x_a.shape[0]}")
    print(f"{'='*60}")

    bs = args.batch_size
    extract_bs = min(bs, x_a.shape[0])

    # Phase 1: Extract environment matrix A from a single batch
    swapper.mode = 'extract_A'
    _ = model(x_a[:extract_bs].to(device))
    print(f"\n[Phase 1] Extracted env A stats (mu shape: {swapper.mu_A.shape})")

    # Phase 2: Extract environment matrix B from a single batch
    swapper.mode = 'extract_B'
    _ = model(x_b[:extract_bs].to(device))
    print(f"[Phase 2] Extracted env B stats (mu shape: {swapper.mu_B.shape})")

    # Phase 3: Baseline accuracy on env A (passthrough mode)
    swapper.mode = 'passthrough'
    out_a = batched_forward(model, x_a, bs, device)
    pred_a = out_a.argmax(dim=1).cpu()
    acc_a = (pred_a == y_a).float().mean().item()
    print(f"[Phase 3] Env A ({args.corruption_a}) baseline accuracy: {acc_a:.2%}")

    # Phase 4: Swap intervention — inject env B's statistics into A's images
    swapper.mode = 'swap_A_to_B'
    out_swap = batched_forward(model, x_a, bs, device)
    pred_swap = out_swap.argmax(dim=1).cpu()
    acc_swap = (pred_swap == y_a).float().mean().item()
    consistency = (pred_a == pred_swap).float().mean().item()

    print(f"\n[Phase 4] After injecting env B matrix into env A images:")
    print(f"          Accuracy: {acc_swap:.2%} (baseline: {acc_a:.2%})")
    print(f"          Prediction consistency (A vs A->B swap): {consistency:.2%}")
    print(f"{'='*60}")

    hook_handle.remove()


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
    parser.add_argument('--hook_layer', type=int, default=1,
                        help='ViT block index to hook (0-based)')
    args = parser.parse_args()
    run_swap_experiment(args)
