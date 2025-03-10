from os import PathLike
import random
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.preprocessing import LabelEncoder
import wandb.wandb_run

def wand_stub():
    return wandb.init(mode="disabled")

def set_seed(seed: int) -> None:
        """Set random seed for all libraries to ensure reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # For reproducible behavior in CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def training_loop(
    model: nn.Module,
    criterion,
    optimizer,
    device,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    num_epochs: int,
    label_encoder: LabelEncoder,
    target_feature: str,
    current_run: wandb.wandb_run.Run,
    validation_name: str,
):
    for epoch in range(num_epochs):
        model.train()
        print(f"\nEpoch {epoch+1}:")
        # Training metrics
        total_training_loss = 0
        training_predictions = []
        training_true_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            encoded_labels = torch.from_numpy(
                label_encoder.transform(labels[target_feature])
            ).to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, encoded_labels)
            loss.backward()
            optimizer.step()

            total_training_loss += loss.item()

            # Collect predictions and true labels
            _, predicted = torch.max(predictions, 1)
            training_predictions.extend(predicted.cpu().numpy())
            training_true_labels.extend(encoded_labels.cpu().numpy())

            # Log batch metrics
            current_run.log(
                {
                    f"batch_loss": loss.item(),
                },
            )

        # Calculate training metrics
        avg_training_loss = total_training_loss / len(train_loader)

        # ========== EPOCH VALIDATION LOOOP ==========
        (validation_accuracies, validation_file_accuracies), avg_validation_loss, (validation_chunk_accuracy, validation_file_accuracy) = (
            test_loop(
                model=model,
                device=device,
                test_loader=validation_loader,
                criterion=criterion,
                label_encoder=label_encoder,
                target_feature=target_feature,
                test_name=validation_name,
            )
        )
        print(f"Training Loss: {avg_training_loss:.4f} | Avg. Validation Loss: {avg_validation_loss:.4f}")
        print(f"Validation Chunk-level Accuracy: {100*validation_chunk_accuracy:.2f}%")
        print(f"Validation File-level Accuracy: {100*validation_file_accuracy:.2f}%")
        print("Logging to wandb")

        current_run.log(
            {
                "epoch": epoch,
                "train_loss": avg_training_loss,
                "validation_loss": avg_validation_loss,
                "validation_chunk_accuracy": validation_chunk_accuracy,
                "validation_file_accuracy": validation_file_accuracy,
            }
        )

    # ======== AFTER TRAINING METRICS ========

    # last validation results
    current_run.log(
        {
            "validation_chunk_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in validation_accuracies.items()],
                columns=["group", "accuracy"],
            ),
            "validation_file_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in validation_file_accuracies.items()],
                columns=["group", "accuracy"],
            ),
        }
    )

    print("Finished training")
    
    
def test_loop(
    model: nn.Module,
    device,
    test_loader: DataLoader,
    criterion,
    label_encoder: LabelEncoder,
    target_feature,
    test_name: str,
):
    model.eval()
    total_test_loss = 0
    chunk_predictions = []
    chunk_true_labels = []
    architecture_chunk_predictions: dict[str, list] = {}
    architecture_chunk_true_labels: dict[str, list] = {}
    
    # For file-level majority voting
    file_predictions_map = {}
    file_true_labels_map = {}
    architecture_file_predictions: dict[str, dict] = {}
    architecture_file_true_labels: dict[str, dict] = {}

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            encoded_labels = torch.from_numpy(
                label_encoder.transform(labels[target_feature])
            ).to(device)

            outputs = model(images)
            loss = criterion(outputs, encoded_labels)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            batch_predictions = predicted.cpu().numpy()
            batch_true_labels = encoded_labels.cpu().numpy()
            
            # Store chunk-level predictions
            chunk_predictions.extend(batch_predictions)
            chunk_true_labels.extend(batch_true_labels)

            # Store predictions by architecture for chunk-level metrics
            for i, arch in enumerate(labels["architecture"]):
                if arch not in architecture_chunk_predictions:
                    architecture_chunk_predictions[arch] = []
                    architecture_chunk_true_labels[arch] = []
                    architecture_file_predictions[arch] = {}
                    architecture_file_true_labels[arch] = {}

                architecture_chunk_predictions[arch].append(batch_predictions[i])
                architecture_chunk_true_labels[arch].append(batch_true_labels[i])

            # Store predictions by file for majority voting
            for i, (pred, true_label, file_path, arch) in enumerate(zip(batch_predictions, batch_true_labels, labels["file_path"], labels["architecture"])):
                if file_path not in file_predictions_map:
                    file_predictions_map[file_path] = []
                    file_true_labels_map[file_path] = true_label
                    architecture_file_predictions[arch][file_path] = []
                    architecture_file_true_labels[arch][file_path] = true_label
                
                file_predictions_map[file_path].append(pred)
                architecture_file_predictions[arch][file_path].append(pred)

    # ====== AFTER TESTING METRICS ======
    
    # Calculate chunk-level accuracies per architecture
    chunk_accuracies = {}
    for arch in architecture_chunk_predictions:
        arch_accuracy = np.mean(
            np.array(architecture_chunk_predictions[arch])
            == np.array(architecture_chunk_true_labels[arch])
        )
        chunk_accuracies[arch] = arch_accuracy
        print(f"{arch} Chunk-level Accuracy: {100*arch_accuracy:.2f}%")

    # Calculate file-level accuracies per architecture using majority voting
    file_accuracies = {}
    for arch in architecture_file_predictions:
        file_preds = []
        file_true = []
        for file_path in architecture_file_predictions[arch]:
            # Get majority vote for this file
            chunk_predictions_for_file = architecture_file_predictions[arch][file_path]
            vote_distribution = np.bincount(chunk_predictions_for_file, minlength=len(label_encoder.classes_))
            file_prediction = vote_distribution.argmax()
            file_true_label = architecture_file_true_labels[arch][file_path]
            
            file_preds.append(file_prediction)
            file_true.append(file_true_label)
        
        file_accuracy = np.mean(np.array(file_preds) == np.array(file_true))
        file_accuracies[arch] = file_accuracy
        print(f"{arch} File-level Accuracy: {100*file_accuracy:.2f}%")

    # Calculate overall metrics
    avg_testing_loss = total_test_loss / len(test_loader)
    # Calculate chunk-level total accuracy as mean of per-architecture accuracies
    chunk_total_accuracy = np.mean(list(chunk_accuracies.values()))
    
    # Calculate overall file-level accuracy
    all_file_predictions = []
    all_file_true_labels = []
    for file_path in file_predictions_map:
        chunk_predictions_for_file = file_predictions_map[file_path]
        vote_distribution = np.bincount(chunk_predictions_for_file, minlength=len(label_encoder.classes_))
        file_prediction = vote_distribution.argmax()
        file_true_label = file_true_labels_map[file_path]
        
        all_file_predictions.append(file_prediction)
        all_file_true_labels.append(file_true_label)
    
    file_total_accuracy = np.mean(np.array(all_file_predictions) == np.array(all_file_true_labels))
    
    print(f"============== FINISHED {test_name} ==============")
    print(f"Chunk-level Total Accuracy: {100*chunk_total_accuracy:.2f}%")
    print(f"File-level Total Accuracy: {100*file_total_accuracy:.2f}%")

    return (chunk_accuracies, file_accuracies), avg_testing_loss, (chunk_total_accuracy, file_total_accuracy)


def save_model_as_onnx(model: nn.Module, model_name: str, sample_dataset: Dataset, save_dir: PathLike = None):
    save_path = f"wandb/{model_name}.onnx" if save_dir is None else f"{save_dir}/{model_name}.onnx"
    original_device = next(model.parameters()).device
    
    try:
        model = model.to('cpu')
        
        sample_input = DataLoader(sample_dataset, batch_size=1, shuffle=False)
        sample_input = next(iter(sample_input))[0]
        
        torch.onnx.export(
            model,
            sample_input,
            save_path,
        )

        wandb.save(save_path, policy="now")

    except Exception as e:
        print(f"Failed to save model as onnx: {e}")
    finally:
        import os
        model = model.to(original_device)
        os.remove(save_path)