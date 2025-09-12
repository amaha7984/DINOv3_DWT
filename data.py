# data.py
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.distributed as dist

class Satellite_Dataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.data = ImageFolder(dir_path, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.496, 0.496, 0.496), std=(0.244, 0.244, 0.244))
    ])

def prepare_dataloader(path, transform, batch_size, is_train=True):
    dataset = Satellite_Dataset(path, transform=transform)

    use_ddp = dist.is_available() and dist.is_initialized()
    sampler = DistributedSampler(
        dataset,
        shuffle=is_train,
        drop_last=is_train
    ) if use_ddp else None

    # shuffle only when not using a sampler
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and is_train),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True if 4 > 0 else False,
        drop_last=is_train
    )
