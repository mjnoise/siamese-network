import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

resize = v2.Resize((64, 64))

TRAIN_TRANSFORM = v2.Compose(
    [
        resize,
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

VAL_TRANSFORM = v2.Compose(
    [resize, v2.ToTensor(), v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)


def load_dataset(
    dataset_path, train_transform=TRAIN_TRANSFORM, val_transform=VAL_TRANSFORM
):
    """"""
    train_dataset = datasets.ImageFolder(
        root=dataset_path + "/sub_train_dataset", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=dataset_path + "/sub_validation_dataset", transform=val_transform
    )
    return (train_dataset, val_dataset)


class TripleDataset(Dataset):
    """
    Custom Dataset that generates triplets (anchor, positive, negative) for
    training the Siamese Network.
     - anchor: An image from the dataset
     - positive: Another image from the same class as the anchor
     - negative: An image from a different class than the anchor
    """
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns Anchor, Positive, Negative samples
        """
        anchor_img, anchor_label = self.data[index]

        # Find a positive sample
        positive_index = index
        while positive_index == index:
            positive_index = torch.randint(0, len(self.data), (1,)).item()
        while self.data[positive_index][1] != anchor_label:
            positive_index = torch.randint(0, len(self.data), (1,)).item()
        positive_img, _ = self.data[positive_index]

        # Find a negative sample
        negative_index = index
        while negative_index == index:
            negative_index = torch.randint(0, len(self.data), (1,)).item()
        while self.data[negative_index][1] == anchor_label:
            negative_index = torch.randint(0, len(self.data), (1,)).item()
        negative_img, _ = self.data[negative_index]

        return (anchor_img, positive_img, negative_img)


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return (train_loader, val_loader)