import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from datetime import timedelta

from model import DINOv3WithDWT
from data import prepare_dataloader, get_transforms
from utils import save_checkpoint, ddp_all_reduce_sum_tensor  


# ---------------------------
# DDP init/finalize (env:// for torchrun)
# ---------------------------
def ddp_init():
    rank        = int(os.environ["RANK"])
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=30),
        world_size=world_size,
        rank=rank,
    )
    return rank, local_rank, world_size


def ddp_finalize():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------
# Train / Eval loops
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss_local, correct_local, total_local = 0.0, 0.0, 0.0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss_local += loss.item() * bs
        preds = torch.argmax(output, dim=1)
        correct_local += (preds == y).sum().item()
        total_local += bs

    tl = ddp_all_reduce_sum_tensor(torch.tensor([total_loss_local], device=device))
    cc = ddp_all_reduce_sum_tensor(torch.tensor([correct_local], device=device))
    tt = ddp_all_reduce_sum_tensor(torch.tensor([total_local], device=device))

    avg_loss = (tl.item() / tt.item()) if tt.item() > 0 else 0.0
    acc = (cc.item() / tt.item() * 100.0) if tt.item() > 0 else 0.0
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    correct_local, total_local, loss_sum_local = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)

            bs = y.size(0)
            loss_sum_local += loss.item() * bs
            preds = torch.argmax(out, dim=1)
            correct_local += (preds == y).sum().item()
            total_local += bs

    ls = ddp_all_reduce_sum_tensor(torch.tensor([loss_sum_local], device=device))
    cc = ddp_all_reduce_sum_tensor(torch.tensor([correct_local], device=device))
    tt = ddp_all_reduce_sum_tensor(torch.tensor([total_local], device=device))

    avg_loss = (ls.item() / tt.item()) if tt.item() > 0 else 0.0
    acc = (cc.item() / tt.item() * 100.0) if tt.item() > 0 else 0.0
    return avg_loss, acc


# ---------------------------
# Main train entrypoint (for torchrun)
# ---------------------------
def main(args):
    rank, local_rank, world_size = ddp_init()
    device = torch.device(f"cuda:{local_rank}")

    # global batch must be divisible by #processes
    assert args.batch_size % world_size == 0, "batch_size must be divisible by world_size"
    batch_per_gpu = args.batch_size // world_size


    train_loader = prepare_dataloader(args.train_path, get_transforms(), batch_per_gpu, is_train=True)
    val_loader   = prepare_dataloader(args.val_path,   get_transforms(), batch_per_gpu, is_train=False)

    model = DINOv3WithDWT(
        num_classes=args.num_classes,
        use_all_bands=True,
        repo_dir=".../dinov3",
        weights=".../weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        freeze_backbone=True
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10

    train_loss_log, val_loss_log, train_acc_log, val_acc_log = [], [], [], []

    for epoch in range(args.epochs):
    
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"[Epoch {epoch + 1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if rank == 0:
            train_loss_log.append(train_loss)
            train_acc_log.append(train_acc)
            val_loss_log.append(val_loss)
            val_acc_log.append(val_acc)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # save by best val acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(model, epoch, optimizer, best_val_acc,
                                os.path.join(args.save_dir, "best_val_acc.pth"))

            # save by best val loss + early stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, epoch, optimizer, best_val_acc,
                                os.path.join(args.save_dir, "best_val_loss.pth"))
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= early_stop_patience:
                print(f"[Early Stopping] No improvement for {early_stop_patience} epochs.")
                break

    # rank 0 plots
    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(train_loss_log, label="Train Loss")
        plt.plot(val_loss_log, label="Val Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.savefig(os.path.join(args.save_dir, "loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(train_acc_log, label="Train Acc")
        plt.plot(val_acc_log, label="Val Acc")
        plt.legend()
        plt.title("Accuracy Curve")
        plt.savefig(os.path.join(args.save_dir, "acc_curve.png"))
        plt.close()

    ddp_finalize()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train")
    parser.add_argument("--val_path", type=str, default="data/val")
    parser.add_argument("--save_dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # launched via torchrun; just call main(args)
    main(args)

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --train_path ... --val_path ... --batch_size 64 --epochs 100
