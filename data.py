from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

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

