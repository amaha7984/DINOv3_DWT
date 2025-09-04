import torch
import os
from torch.distributed import init_process_group, destroy_process_group

def set_device():
    return torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, epoch, optimizer, best_acc, path):
    state = {
        'epoch': epoch + 1,
        'model': model.module.state_dict(),
        'best_accuracy': best_acc,
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, path)

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)