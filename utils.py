# utils.py
import os
import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    return get_global_rank() == 0


def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def save_checkpoint(model, epoch, optimizer, best_acc, path: str):
    """Rank-agnostic saver; caller should gate with is_main_process()."""
    state = {
        "epoch": epoch + 1,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "best_accuracy": best_acc,
        "optimizer": optimizer.state_dict(),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def ddp_all_reduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    """Sum-reduce a tensor across all processes; returns the same tensor."""
    if is_dist_avail_and_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def set_device():
   
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")
