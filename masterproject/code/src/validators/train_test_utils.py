from os import PathLike
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.preprocessing import LabelEncoder

def wand_stub():
    return wandb.init(mode="disabled")


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
    wandb_run: wandb.wandb_run.Run,
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
            
            wandb_run.log(
                {
                    f"batch_loss": loss.item(),
                },
            )

        # Calculate training metrics
        avg_training_loss = total_training_loss / len(train_loader)

        # ========== EPOCH VALIDATION LOOOP ==========
        validation_accuracies, avg_validation_loss, validation_total_accuracy = (
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
        print(f"Validation Total Accuracy: {100*validation_total_accuracy:.2f}%")
        print("Logging to wandb")

        wandb_run.log(
            {
                "epoch": epoch,
                "train_loss": avg_training_loss,
                "validation_loss": avg_validation_loss,
                "validation_accuracy": validation_total_accuracy,
            }
        )

    # ======== AFTER TRAINING METRICS ========

    # last validation results
    wandb_run.log(
        {
            "validation_accuracy_per_group": wandb.Table(
                data=[[group, acc] for group, acc in validation_accuracies.items()],
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
    testing_predictions = []
    testing_true_labels = []
    architecture_predictions: dict[str, list] = {}
    architecture_true_labels: dict[str, list] = {}

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
            testing_predictions.extend(predicted.cpu().numpy())
            testing_true_labels.extend(encoded_labels.cpu().numpy())

            for i, arch in enumerate(labels["architecture"]):
                if arch not in architecture_predictions:
                    architecture_predictions[arch] = []
                    architecture_true_labels[arch] = []

                architecture_predictions[arch].append(predicted[i].cpu().item())
                architecture_true_labels[arch].append(encoded_labels[i].cpu().item())

    # ====== AFTER TESTING METRICS ======
    testing_accuracies = {}
    for arch in architecture_predictions:
        arch_accuracy = np.mean(
            np.array(architecture_predictions[arch])
            == np.array(architecture_true_labels[arch])
        )
        testing_accuracies[arch] = arch_accuracy
        print(f"{arch} Accuracy: {100*arch_accuracy:.2f}%")

    avg_testing_loss = total_test_loss / len(test_loader)
    testing_total_accuracy = np.mean(
        np.array(testing_predictions) == np.array(testing_true_labels)
    )
    print(f"============== FINISHED {test_name} ==============")

    return testing_accuracies, avg_testing_loss, testing_total_accuracy


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