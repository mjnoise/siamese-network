import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from backbone_classifier import Classifier
from dataset import TripleDataset, load_dataset
from siamese_model import SiameseModel
from utils import (
    get_embeddings,
    get_image,
    get_query_img_embedding,
    siamese_training_loop,
    training_loop,
    compute_class_weights,
    find_closest,
)

random_seed = 42
torch.manual_seed(random_seed)


def main():
    # Configuration
    dataset_path = "clothing-dataset-small"
    batch_size = 32
    num_workers = 2
    n_epochs = 5
    lr = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load train/validation datasets (ImageFolder)
    train_dataset, val_dataset = load_dataset(dataset_path)
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Model, loss, optimizer, scheduler
    model = Classifier(num_classes=num_classes)
    class_weights = compute_class_weights(train_loader.dataset)
    loss_fcn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.1)

    # Train
    trained_model = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fcn=loss_fcn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        n_epochs=n_epochs,
    )

    # Save best model weights
    torch.save(trained_model.state_dict(), "classifier_best.pth")
    print("Saved trained classifier to classifier_best.pth")

    # Siamese model training
    triple_train_dataset = TripleDataset(train_dataset)

    triple_train_loader = DataLoader(
        triple_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    siamese_model = SiameseModel(backbone=trained_model.backbone)
    siamese_loss_fcn = nn.TripletMarginLoss(margin=1.0, p=2)
    siamese_optimizer = torch.optim.Adam(siamese_model.parameters(), lr=lr)

    trained_siamese_model = siamese_training_loop(
        model=siamese_model,
        train_loader=triple_train_loader,
        loss_fcn=siamese_loss_fcn,
        optimizer=siamese_optimizer,
        device=device,
        n_epochs=n_epochs,
    )

    random_idx = torch.randint(0, len(val_dataset), (1,)).item()
    query_image, query_label = get_image(val_dataset, random_idx)
    val_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )

    query_image_embedding = get_query_img_embedding(
        trained_siamese_model.encoder, val_transform, query_image, device
    )

    embeddings = get_embeddings(
        trained_siamese_model.encoder, val_dataset, device)
    closest_indices = find_closest(embeddings, query_image_embedding)
    print(f"Query image label: {val_dataset.classes[query_label]}")

    print(f"\nDisplaying the {5} most similar items found in the catalog:")
    for idx_c in closest_indices:
        img_c, label_idx_c = get_image(
            val_dataset, idx_c
        )
        label_c = val_dataset.classes[label_idx_c]
        print(f"Retrieved Item - Class: {label_c} (Index: {idx_c})")
        # Display the retrieved image
        plt.imshow(img_c)
        plt.title(f"Retrieved Item - Class: {label_c} (Index: {idx_c})")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
