import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path
import math

from robustbench.data import load_cifar10c, PREPROCESSINGS
from robustbench.loaders import CustomImageFolder


class FeatureSwapper(nn.Module):
    def __init__(self, cls_only=True):
        super().__init__()
        self.mode = 'passthrough'
        self.mu_target = None
        self.sigma_target = None
        self.cls_only = cls_only

    def set_target(self, mu, sigma):
        self.mu_target = mu
        self.sigma_target = sigma

    def forward(self, module, input, output):
        if self.mode == 'passthrough' or self.mu_target is None:
            return output
        if self.cls_only:
            f = output[:, 0:1]
            cur_mu = f.mean(dim=(0, 1), keepdim=True)
            cur_sigma = f.std(dim=(0, 1), keepdim=True)
            f_norm = (f - cur_mu) / (cur_sigma + 1e-6)
            f_swapped = f_norm * self.sigma_target + self.mu_target
            output = output.clone()
            output[:, 0:1] = f_swapped
        else:
            cur_mu = output.mean(dim=(0, 1), keepdim=True)
            cur_sigma = output.std(dim=(0, 1), keepdim=True)
            f_norm = (output - cur_mu) / (cur_sigma + 1e-6)
            output = f_norm * self.sigma_target + self.mu_target
        return output


class MemoryBank:
    def __init__(self, max_size=10, thr_dist=2.0, ema_alpha=0.9):
        self.max_size = max_size
        self.thr_dist = thr_dist
        self.ema_alpha = ema_alpha
        self.entries = []

    def update(self, mu, sigma):
        mu_cpu = mu.detach().cpu().clone()
        sigma_cpu = sigma.detach().cpu().clone()

        if not self.entries:
            self.entries.append((mu_cpu, sigma_cpu))
            return

        dists = [torch.norm(mu_cpu - e[0], p=2).item() for e in self.entries]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)

        if min_dist > self.thr_dist:
            if len(self.entries) >= self.max_size:
                self.entries.pop(0)
            self.entries.append((mu_cpu, sigma_cpu))
        else:
            a = self.ema_alpha
            e = self.entries[min_idx]
            self.entries[min_idx] = (
                a * e[0] + (1 - a) * mu_cpu,
                a * e[1] + (1 - a) * sigma_cpu,
            )

    def sample(self):
        if not self.entries:
            return None, None
        idx = torch.randint(0, len(self.entries), (1,)).item()
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)


def get_block(model, idx):
    if hasattr(model, 'blocks'):
        return model.blocks[idx]
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        return model.encoder.layers[idx]
    raise AttributeError("Cannot find transformer blocks")


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
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_paired_imagenetc(n_examples, severity, data_dir, corruption_a,
                          corruption_b, batch_size=32):
    transform = PREPROCESSINGS['Res256Crop224']
    path_a = Path(data_dir) / corruption_a / str(severity)
    path_b = Path(data_dir) / corruption_b / str(severity)
    da = CustomImageFolder(str(path_a), transform)
    db = CustomImageFolder(str(path_b), transform)
    n = min(n_examples, len(da))
    la = torch.utils.data.DataLoader(da, batch_size=batch_size, shuffle=False)
    lb = torch.utils.data.DataLoader(db, batch_size=batch_size, shuffle=False)
    xa_list, ya_list, xb_list, yb_list = [], [], [], []
    cnt = 0
    for (xa, ya, _), (xb, yb, _) in zip(la, lb):
        xa_list.append(xa); ya_list.append(ya)
        xb_list.append(xb); yb_list.append(yb)
        cnt += xa.shape[0]
        if cnt >= n: break
    xa = torch.cat(xa_list, 0)[:n]; ya = torch.cat(ya_list, 0)[:n]
    xb = torch.cat(xb_list, 0)[:n]; yb = torch.cat(yb_list, 0)[:n]
    assert torch.equal(ya, yb), "Label mismatch"
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    xa = (xa - mean) / std; xb = (xb - mean) / std
    print(f"[*] Paired ImageNet-C: {n} images")
    return xa, ya, xb, yb


def load_paired_cifar10c(n_examples, severity, data_dir, corruption_a,
                         corruption_b, resize=384):
    xa, ya = load_cifar10c(n_examples, severity, data_dir,
                           corruptions=[corruption_a])
    xb, yb = load_cifar10c(n_examples, severity, data_dir,
                           corruptions=[corruption_b])
    assert torch.equal(ya, yb), "Label mismatch"
    xa = F.interpolate(xa, size=(resize, resize), mode='bilinear',
                       align_corners=False)
    xb = F.interpolate(xb, size=(resize, resize), mode='bilinear',
                       align_corners=False)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    xa = (xa - mean) / std; xb = (xb - mean) / std
    return xa, ya, xb, yb


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
def collect_block_features(model, block, x, batch_size, device, cls_only=True):
    features = []
    def hook(module, input, output):
        f = output[:, 0:1] if cls_only else output
        features.append(f.detach().cpu())
        return output
    handle = block.register_forward_hook(hook)
    for i in range(0, x.shape[0], batch_size):
        end = min(i + batch_size, x.shape[0])
        _ = model(x[i:end].to(device))
    handle.remove()
    all_f = torch.cat(features, dim=0)
    return all_f.mean(dim=(0, 1), keepdim=True), all_f.std(dim=(0, 1), keepdim=True)


@torch.no_grad()
def warmup_bank(model, block, x_warmup, batch_size, device, bank, cls_only=True):
    swapper = FeatureSwapper(cls_only=cls_only)
    handle = block.register_forward_hook(swapper)
    for i in range(0, min(x_warmup.shape[0], bank.max_size * batch_size),
                   batch_size):
        end = min(i + batch_size, x_warmup.shape[0])
        xb = x_warmup[i:end].to(device)
        features = []
        def hook(m, inp, out):
            f = out[:, 0:1] if cls_only else out
            features.append(f.detach().cpu())
            return out
        h = block.register_forward_hook(hook)
        _ = model(xb)
        h.remove()
        if features:
            f = torch.cat(features, 0)
            mu = f.mean(dim=(0, 1), keepdim=True)
            sigma = f.std(dim=(0, 1), keepdim=True)
            bank.update(mu, sigma)
    handle.remove()
    print(f"    Bank: {len(bank)} environments")


def configure_lara(model, hook_layer_idx, aux_layer_idx, device, freeze_below=True):
    blocks = []
    if hasattr(model, 'blocks'):
        blocks = model.blocks
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        blocks = model.encoder.layers

    n = len(blocks)
    hook_layer_idx = min(hook_layer_idx, n - 1)

    trainable_params = []
    for i, block in enumerate(blocks):
        for m in block.modules():
            if isinstance(m, nn.LayerNorm):
                if freeze_below and i < hook_layer_idx:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
                else:
                    m.weight.requires_grad = True
                    m.bias.requires_grad = True
                    trainable_params.extend([m.weight, m.bias])

    hook_layer = blocks[hook_layer_idx]
    aux_layer = blocks[min(aux_layer_idx, n - 1)]

    dim = 768
    try:
        dim = (model.heads.head if hasattr(model, 'heads') else model.head).in_features
    except:
        pass
    num_classes = 1000 if not hasattr(model, 'heads') else (
        model.heads.head.out_features if hasattr(model.heads, 'head') else 1000)
    try:
        if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
            num_classes = model.heads.head.out_features
        elif hasattr(model, 'head'):
            num_classes = model.head.out_features
    except:
        pass

    aux_head = nn.Linear(dim, num_classes).to(device)
    trainable_params.extend([aux_head.weight, aux_head.bias])

    return trainable_params, hook_layer, aux_layer, aux_head


def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def run_lara(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device}")

    model = build_model(args, device)
    x_a, y_a, x_b, y_b = load_data(args)

    trainable_params, hook_layer, aux_layer, aux_head = configure_lara(
        model, args.hook_layer, args.aux_layer, device)
    print(f"[*] Trainable: {len(trainable_params)} params "
          f"(LN block[{args.hook_layer}-11] + aux_head@{args.aux_layer})")

    bank = MemoryBank(max_size=args.bank_size, thr_dist=args.thr_d)
    cls_only = not args.all_tokens
    warmup_bank(model, hook_layer, x_b, args.batch_size, device, bank, cls_only)

    swapper = FeatureSwapper(cls_only=cls_only)
    swapper_handle = hook_layer.register_forward_hook(swapper)

    aux_cls = []
    def aux_hook(module, input, output):
        aux_cls.append(output[:, 0])
        return output
    aux_handle = aux_layer.register_forward_hook(aux_hook)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    bs = args.batch_size
    n = x_a.shape[0]
    correct = 0
    total_base = 0.0
    total_cons = 0.0
    steps = 0

    for i in range(0, n, bs):
        end = min(i + bs, n)
        x_batch = x_a[i:end].to(device)

        for _ in range(args.inner_steps):
            optimizer.zero_grad()

            swapper.mode = 'passthrough'
            aux_cls.clear()
            logits_orig = model(x_batch)
            cls_aux_orig = aux_cls.pop() if aux_cls else None
            logits_aux_orig = aux_head(cls_aux_orig) if cls_aux_orig is not None else None

            p_orig = F.softmax(logits_orig, dim=-1)
            h_orig = softmax_entropy(logits_orig)
            loss_base = h_orig.mean()
            if logits_aux_orig is not None:
                loss_base += 0.5 * softmax_entropy(logits_aux_orig).mean()
            total_base += loss_base.item()

            loss_cons = torch.tensor(0.0, device=device)
            hist_mu, hist_sigma = bank.sample()

            if hist_mu is not None and args.beta > 0:
                swapper.mode = 'swap'
                swapper.set_target(hist_mu.to(device), hist_sigma.to(device))
                aux_cls.clear()
                logits_swap = model(x_batch)
                cls_aux_swap = aux_cls.pop() if aux_cls else None
                logits_aux_swap = aux_head(cls_aux_swap) if cls_aux_swap is not None else None

                p_swap = F.softmax(logits_swap, dim=-1)
                h_swap = softmax_entropy(logits_swap)
                delta_h = (h_swap - h_orig.detach()).mean().item()
                w_affinity = float(math.exp(-max(delta_h, 0) / 1.0))

                if w_affinity > 0.2:
                    log_p_swap = F.log_softmax(logits_swap, dim=-1)
                    loss_cons = w_affinity * F.kl_div(
                        log_p_swap, p_orig.detach(), reduction='batchmean')
                    if logits_aux_swap is not None:
                        log_p_aux_swap = F.log_softmax(logits_aux_swap, dim=-1)
                        p_aux_orig = F.softmax(logits_aux_orig.detach(), dim=-1)
                        loss_cons += w_affinity * F.kl_div(
                            log_p_aux_swap, p_aux_orig, reduction='batchmean')
                    total_cons += loss_cons.item()

                swapper.mode = 'passthrough'

            total_loss = loss_base + args.beta * loss_cons
            total_loss.backward()
            optimizer.step()
            steps += 1

        with torch.no_grad():
            out = model(x_batch)
            correct += (out.argmax(1).cpu() == y_a[i:end]).sum().item()

        with torch.no_grad():
            swapper.mode = 'passthrough'
            mu, sigma = collect_block_features(model, hook_layer, x_batch,
                                                1, device, cls_only)
            bank.update(mu, sigma)

    swapper_handle.remove()
    aux_handle.remove()
    acc = correct / n
    print(f"\n{'='*60}")
    print(f" LARA-Aux Results")
    print(f" Hook: block[{args.hook_layer}] | Aux: block[{args.aux_layer}]")
    print(f" beta={args.beta} | lr={args.lr}")
    print(f" Accuracy: {acc:.2%} | base: {total_base/steps:.4f} |"
          f" cons: {total_cons/steps:.4f} | bank: {len(bank)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LARA-TTA')
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['imagenet', 'cifar10'])
    parser.add_argument('--data_dir', type=str,
                        default='/root/data/picture/ImageNet-C')
    parser.add_argument('--corruption_a', type=str, default='defocus_blur')
    parser.add_argument('--corruption_b', type=str, default='contrast')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hook_layer', type=int, default=5)
    parser.add_argument('--aux_layer', type=int, default=9,
                        help='Auxiliary classifier layer (for short gradient path)')
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--inner_steps', type=int, default=3)
    parser.add_argument('--bank_size', type=int, default=10)
    parser.add_argument('--thr_d', type=float, default=2.0)
    parser.add_argument('--all_tokens', action='store_true')
    parser.add_argument('--backend', type=str, default='torchvision',
                        choices=['torchvision', 'timm'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--no_pretrained', action='store_false', dest='pretrained')
    args = parser.parse_args()
    run_lara(args)
