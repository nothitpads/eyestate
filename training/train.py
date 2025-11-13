import os
import argparse
from pathlib import Path
import time
import copy
import csv
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torchvision

import sys
TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from preprocess import get_dataloaders

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn=True, relu=True):
        super().__init__()
        layers = [nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_planes))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(4, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels)
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2,3))    # (B, C)
        max_pool,_ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)   # (B, C)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        scale = torch.sigmoid(out).unsqueeze(-1).unsqueeze(-1)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = BasicConv(2, 1, kernel_size=7, padding=3, bn=False, relu=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max,_ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, max], dim=1)
        scale = torch.sigmoid(self.compress(cat))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate(in_channels, reduction=reduction)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


# Model builder (EfficientNet-V2 + CBAM)
def build_model(num_classes=2, cbam=True, pretrained=True):
    model = torchvision.models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None)
    in_features = model.classifier[1].in_features
    
    # replace classifier head !!!!!!!!!!
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    # attach CBAM after features if requested
    if cbam:
        class WithCBAM(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.features = base.features
                self.cbam = CBAM(in_channels=in_features)
                self.classifier = base.classifier

            def forward(self, x):
                x = self.features(x)
                # x shape: B x C x H x W ; EfficientNet's final conv channels == in_features
                x = self.cbam(x)
                x = torch.flatten(torch.mean(x, dim=[2,3]), 1)  # global avg pool (safe)
                x = self.classifier(x)
                return x
        model = WithCBAM(model)
    return model

# Utility metrics
def compute_metrics(outputs, targets):
    # outputs: logits tensor (B, C)
    preds = torch.argmax(outputs, dim=1).detach().cpu()
    targets = targets.detach().cpu()
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    # per-class counts
    num_classes = outputs.size(1)
    tp = [0]*num_classes
    fp = [0]*num_classes
    fn = [0]*num_classes
    for c in range(num_classes):
        tp[c] = int(((preds == c) & (targets == c)).sum().item())
        fp[c] = int(((preds == c) & (targets != c)).sum().item())
        fn[c] = int(((preds != c) & (targets == c)).sum().item())
    return correct, total, tp, fp, fn

def aggregate_and_report(epoch, phase, acc_count, total_count, tp_sum, fp_sum, fn_sum, classes):
    acc = acc_count / total_count if total_count>0 else 0.0
    per_class_f1 = {}
    for i,cname in enumerate(classes):
        prec = tp_sum[i] / (tp_sum[i] + fp_sum[i]) if (tp_sum[i] + fp_sum[i])>0 else 0.0
        rec  = tp_sum[i] / (tp_sum[i] + fn_sum[i]) if (tp_sum[i] + fn_sum[i])>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        per_class_f1[cname] = {"precision": prec, "recall": rec, "f1": f1}
    # brief summary dict
    summary = {
        "epoch": epoch,
        "phase": phase,
        "accuracy": acc,
        "per_class": per_class_f1
    }
    return summary

# Grad-CAM utility (simple)
def gradcam_visualize(model, input_tensor, target_class=None, device="cpu"):
    model.eval()
    input_tensor = input_tensor.to(device)
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    target_module = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Conv2d):
            target_module = m
            break
    if target_module is None:
        raise RuntimeError("No Conv2d found for Grad-CAM hook.")

    h_fwd = target_module.register_forward_hook(forward_hook)
    h_bwd = target_module.register_backward_hook(backward_hook)

    logits = model(input_tensor)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    loss = logits[:, target_class].sum()
    model.zero_grad()
    loss.backward()

    act = activations[-1].cpu()[0]       # C x H x W
    grad = gradients[-1].cpu()[0]        # C x H x W
    weights = grad.mean(dim=(1,2), keepdim=True)  # C x 1 x 1
    cam = (weights * act).sum(dim=0)  # H x W
    cam = torch.relu(cam)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    cam_np = cam.numpy()
    h_fwd.remove()
    h_bwd.remove()
    return cam_np

# Training
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")

    # dataloaders
    dl_train, dl_val, dl_test, meta = get_dataloaders(
        data_dir=args.data_dir,
        classes=["open", "close"],
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
        num_workers=args.num_workers,
        pin_memory=True,
        seed=args.seed,
        save_manifests=True,
    )
    classes = meta["classes"]

    model = build_model(num_classes=len(classes), cbam=True, pretrained=True)
    model = model.to(device)

    # freeze features initially
    all_feature_params = []
    # freeze everything except classifier initially
    for name, p in model.named_parameters():
        if "classifier" in name or "cbam" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
        all_feature_params.append((name, p.requires_grad))

    # optimizer uses only params with requires_grad=True
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.use_amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.freeze_epochs))

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Unfreeze stage (unchanged)
        if epoch == args.freeze_epochs + 1:
            try:
                children = list(model.features.children())
                unfreeze_k = args.unfreeze_blocks
                for child in children[-unfreeze_k:]:
                    for p in child.parameters():
                        p.requires_grad = True
                for name, p in model.named_parameters():
                    if "classifier" in name or "cbam" in name:
                        p.requires_grad = True
                optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr * 0.5, weight_decay=args.weight_decay)
                scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch))
                print(f"Epoch {epoch}: Unfroze last {unfreeze_k} blocks and reinitialized optimizer.")
            except Exception:
                for p in model.parameters():
                    p.requires_grad = True
                optimizer = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
                scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch))
                print(f"Epoch {epoch}: Unfroze all parameters (fallback).")
    
        # training phase with progress bar
        model.train()
        train_correct = 0
        train_total = 0
        tp_sum = [0] * len(classes)
        fp_sum = [0] * len(classes)
        fn_sum = [0] * len(classes)
        running_loss = 0.0
        batch_count = 0
    
        pbar = tqdm(enumerate(dl_train, 1), total=len(dl_train), desc=f"Epoch {epoch}/{args.epochs} [train]", ncols=120)
        for batch_idx, (xb, yb) in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=args.use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            # metrics
            c, t, tp, fp, fn = compute_metrics(logits, yb)
            train_correct += c
            train_total += t
            for i in range(len(classes)):
                tp_sum[i] += tp[i]
                fp_sum[i] += fp[i]
                fn_sum[i] += fn[i]
    
            running_loss += loss.item()
            batch_count += 1
    
            # display values in progress bar
            avg_loss = running_loss / batch_count
            batch_acc = train_correct / train_total if train_total > 0 else 0.0
            # get current LR
            current_lr = optimizer.param_groups[0]['lr'] if len(optimizer.param_groups) > 0 else 0.0
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "batch_acc": f"{batch_acc:.4f}",
                "lr": f"{current_lr:.2e}"
            })
    
        pbar.close()
        train_summary = aggregate_and_report(epoch, "train", train_correct, train_total, tp_sum, fp_sum, fn_sum, classes)
    
        # validation phase with progress bar
        model.eval()
        val_correct = 0
        val_total = 0
        tp_sum = [0] * len(classes)
        fp_sum = [0] * len(classes)
        fn_sum = [0] * len(classes)
    
        pbar_val = tqdm(enumerate(dl_val, 1), total=len(dl_val), desc=f"Epoch {epoch}/{args.epochs} [val]", ncols=120)
        with torch.no_grad():
            for batch_idx, (xb, yb) in pbar_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with autocast(enabled=args.use_amp):
                    logits = model(xb)
                c, t, tp, fp, fn = compute_metrics(logits, yb)
                val_correct += c
                val_total += t
                for i in range(len(classes)):
                    tp_sum[i] += tp[i]
                    fp_sum[i] += fp[i]
                    fn_sum[i] += fn[i]
                # optional: show running val accuracy
                running_val_acc = val_correct / val_total if val_total > 0 else 0.0
                pbar_val.set_postfix({"val_acc": f"{running_val_acc:.4f}"})
        pbar_val.close()
    
        val_summary = aggregate_and_report(epoch, "val", val_correct, val_total, tp_sum, fp_sum, fn_sum, classes)
    
        # scheduler step
        try:
            scheduler.step()
        except Exception:
            pass
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{args.epochs}  train_acc={train_summary['accuracy']:.4f}  val_acc={val_summary['accuracy']:.4f}  time={epoch_time:.1f}s")
        entry = {"epoch": epoch, "train_acc": train_summary["accuracy"], "val_acc": val_summary["accuracy"], "time": epoch_time}
        history.append(entry)
    
        # save best
        if val_summary["accuracy"] > best_val_acc:
            best_val_acc = val_summary["accuracy"]
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "classes": classes,
            }
            best_path = Path(args.save_dir) / "best_model.pth"
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(best_state, str(best_path))
            print(f"Saved best model (val_acc={best_val_acc:.4f}) -> {best_path}")
    
        # write epoch history to CSV
        history_path = Path(args.save_dir) / "train_history.csv"
        header = ["epoch", "train_acc", "val_acc", "time"]
        write_header = not history_path.exists()
        with open(history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(entry)

    # final test evaluation using best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    model.eval()
    test_correct = 0
    test_total = 0
    tp_sum = [0] * len(classes)
    fp_sum = [0] * len(classes)
    fn_sum = [0] * len(classes)
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            c, t, tp, fp, fn = compute_metrics(logits, yb)
            test_correct += c
            test_total += t
            for i in range(len(classes)):
                tp_sum[i] += tp[i]
                fp_sum[i] += fp[i]
                fn_sum[i] += fn[i]
    test_summary = aggregate_and_report(epoch, "test", test_correct, test_total, tp_sum, fp_sum, fn_sum, classes)
    print("Final test accuracy:", test_summary["accuracy"])
    # save final state (best + final metrics)
    final_out = {
        "best_val_acc": best_val_acc,
        "test_acc": test_summary["accuracy"],
        "history": history,
    }
    with open(Path(args.save_dir) / "train_results.txt", "w") as f:
        f.write(str(final_out))
    print("Training complete.")

# -------------------------
# Argparse
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train eye-state EfficientNetV2+CBAM")
    p.add_argument("--data-dir", type=str, default="../data/raw", help="path to raw data root (contains train/.. folders)")
    p.add_argument("--image-size", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--freeze-epochs", type=int, default=5, help="number of epochs to keep feature extractor frozen")
    p.add_argument("--unfreeze-blocks", type=int, default=2, help="number of last feature blocks to unfreeze after freeze period")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-pct", type=float, default=0.1)
    p.add_argument("--test-pct", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="../weights")
    p.add_argument("--use-amp", action="store_true", help="use mixed precision")
    p.add_argument("--force-cpu", action="store_true", help="disable CUDA even if available")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
