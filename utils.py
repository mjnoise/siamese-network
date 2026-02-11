from __future__ import annotations
import copy
from collections import Counter
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassAccuracy


def training_loop(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fcn: Callable,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    n_epochs: int,
):
    """Training loop for the backbone classifier model using cross-entropy loss and accuracy metric."""

    model.to(device)

    n_classes = len(train_loader.dataset.classes)

    train_acc_metric = MulticlassAccuracy(num_classes=n_classes).to(device)
    val_acc_metric = MulticlassAccuracy(num_classes=n_classes).to(device)

    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None

    print("-------------Starting training------------")

    for epoch in range(n_epochs):
        model.train()
        running_train_loss = 0.0
        train_acc_metric.reset()

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training", leave=False
        )

        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)

            train_acc_metric.update(preds, labels)
            train_pbar.set_postfix(batch_loss=loss.item())

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = train_acc_metric.compute().item()

        model.eval()
        running_val_loss = 0.0
        val_acc_metric.reset()

        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Validation", leave=False
        )

        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = loss_fcn(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)

                val_acc_metric.update(preds, labels)
                val_pbar.set_postfix(batch_loss=loss.item())

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = val_acc_metric.compute().item()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Step the scheduler based on validation loss
                scheduler.step(epoch_val_loss)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch+1}/{n_epochs} - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"-------------Training complete. Best val acc: {best_val_acc:.4f} at epoch {best_epoch}------------"
        )

        if best_model_state is not None:
            print(
                f"New best model found at epoch {best_epoch} with val acc: {best_val_acc:.4f}"
            )
            model.load_state_dict(best_model_state)
        else:
            print(
                "Warning: No best model found during training. Returning the last model state."
            )

    return model


def compute_class_weights(dataset):
    """
    Computes class weights inversely proportional to class frequencies.

    Args:
        dataset (torch.utils.data.Dataset): A PyTorch dataset, expected to have
                                             a 'targets' attribute (like torchvision ImageFolder)
                                             or provide labels when iterated.

    Returns:
        torch.Tensor: A tensor containing the weight for each class.
    """
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels
    else:
        print(
            "Dataset has no 'targets' or 'labels' attribute, iterating to get labels..."
        )
        labels = [label for _, label in dataset]

    class_counts = Counter(labels)

    sorted_counts = [class_counts[i] for i in sorted(class_counts)]

    total_samples = len(labels)
    num_classes = len(sorted_counts)

    weights = []
    for count in sorted_counts:
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    return weights_tensor


def siamese_training_loop(
    model,
    train_loader: DataLoader,
    loss_fcn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int,
):
    """Training loop for the Siamese Network using triplet loss."""

    model.to(device)

    best_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    print("-------------Starting Siamese training------------")

    for epoch in range(n_epochs):
        model.train()

        running_train_loss = 0.0
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{n_epochs} - Training", leave=False
        )

        for anchor, positive, negative in train_pbar:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            optimizer.zero_grad()
            anchor_rep, positive_rep, negative_rep = model(anchor, positive, negative)
            loss = loss_fcn(anchor_rep, positive_rep, negative_rep)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            train_pbar.set_postfix(
                batch_loss=f"{running_train_loss / (train_pbar.n + 1):.4f}"
            )

        epoch_loss = running_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs} finished, Average Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            print(
                f"    -> New best model saved (Epoch {best_epoch}, Loss: {best_loss:.4f})"
            )

    print("\n--- Siamese Training Complete ---")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(
            f"Loaded best model weights from Epoch {best_epoch} with lowest loss: {best_loss:.4f}"
        )
    else:
        print(
            "Warning: No best model state was saved (check training loop). Using weights from the last epoch."
        )

    return model


def get_query_img_embedding(encoder, transform, img, device):
    """
    Generates an embedding vector for a single query PIL image.

    Args:
        encoder (nn.Module): The trained embedding model (e.g., SiameseEncoder).
        transform (callable): The torchvision transforms to apply (e.g., resize, normalize).
        img (PIL.Image): The input query image.
        device (torch.device): The device ('cuda' or 'cpu') to perform inference on.

    Returns:
        np.ndarray: The embedding vector as a NumPy array.
    """
    tensor_img = transform(img)
    query_img_tensor = tensor_img.unsqueeze(0).to(device)

    encoder.eval()
    with torch.no_grad():
        query_img_embedding = encoder(query_img_tensor)

    query_img_embedding_np = query_img_embedding.cpu().numpy()

    return query_img_embedding_np


def get_embeddings(model_representation, labeled_dataset, device):
    """
    Computes and returns the feature embeddings for a given dataset using a model.

    Args:
        model_representation: The feature extraction model.
        labeled_dataset: The dataset containing the data samples.
        device: The computational device (e.g., 'cpu' or 'cuda').

    Returns:
        A NumPy array containing the computed embeddings for the entire dataset.
    """
    dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=False)
    model_representation.eval()
    embeddings = []

    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            embedding_img = model_representation.forward(img)
            embeddings.append(embedding_img.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def find_closest(embeddings, target_embedding, num_samples=5):
    """
    Finds the indices of the samples with the closest embeddings to the target embedding.

    Args:
        embeddings: A NumPy array containing the feature embeddings of all samples.
        target_embedding: A NumPy array representing the embedding of the query sample.
        num_samples: The number of closest sample indices to retrieve.

    Returns:
        A NumPy array of the indices corresponding to the closest samples.
    """
    distances = np.linalg.norm(embeddings - target_embedding, axis=1)

    closest_indices = np.argsort(distances)[1 : num_samples + 1]

    return closest_indices


def get_image(dataset, index):
    """
    Retrieves an original image and its label from a dataset by index.

    Args:
        dataset: The dataset object (e.g., a torchvision ImageFolder)
                 that contains a 'samples' list of (path, label) tuples.
        index: The index of the item to retrieve.

    Returns:
        A tuple containing the PIL Image object (in RGB format) and
        its corresponding label.
    """
    path, label = dataset.samples[index]
    original_img = Image.open(path).convert("RGB")
    return original_img, label
