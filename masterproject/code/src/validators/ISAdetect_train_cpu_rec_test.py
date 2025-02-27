from datetime import datetime
import numpy as np
from sklearn.calibration import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import random_split
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch
import wandb.wandb_run
from src.validators.train_test_utils import training_loop, test_loop, save_model_as_onnx


def ISAdetect_train_cpu_rec_test(
    config,
    model_class: nn.Module,
    ISAdetectDataset: Dataset,
    CpuRecDataset: Dataset,
    device,
):
    assert (
        "validation_split" in config["validator"]
    ), 'validation_split not in config["validator"]'
    assert "test_split" in config["validator"], 'test_split not in config["validator"]'

    wandb_group_name = f"{config["validator"]["name"]} {config["model"]["name"]} {config["target_feature"]} {datetime.now().strftime('%H:%M:%S, %d-%m-%Y')}"
    wandb_project = config["validator"]["wandb_project_name"]
    if not wandb_project:
        print("Wandb project name not provided, skipping wandb logging")

    wandb_run = wandb.init(
        project=wandb_project,
        name="ISAdetect_train",
        group=wandb_group_name,
        config=config,
        mode=(
            "online" if config["validator"]["wandb_project_name"] else "disabled"
        ),  # disabled = no-op
    )

    wandb.define_metric("epoch")
    wandb.define_metric("validation_loss", step_metric="epoch")
    wandb.define_metric("validation_loss", step_metric="epoch")
    wandb.define_metric("validation_accuracy", step_metric="epoch")

    groups = list(map(lambda x: x["architecture"], ISAdetectDataset.metadata))
    target_features = list(
        map(lambda x: x[config["target_feature"]], ISAdetectDataset.metadata)
    )

    print(f"group: {set(groups)}")
    print(f"target_feature: {set(target_features)}")

    validation_split = config["validator"]["validation_split"]
    test_split = config["validator"]["test_split"]
    train_split = 1 - validation_split - test_split

    train_size = int(train_split * len(ISAdetectDataset))
    validation_size = int(validation_split * len(ISAdetectDataset))
    test_size = len(ISAdetectDataset) - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = random_split(
        ISAdetectDataset, [train_size, validation_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    model = model_class(**config["model"]["params"])
    model.to(device)
    wandb_run.watch(model)
    criterion = (
        getattr(model, "criterion", None)
        or getattr(nn, config["training"]["criterion"])()
    )
    optimizer = getattr(torch.optim, config["training"]["optimizer"])(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    all_train_labels = [
        ISAdetectDataset.metadata[i][config["target_feature"]]
        for i in range(len(ISAdetectDataset))
    ]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_train_labels)

    EPOCHS = config["training"]["epochs"]
    model.train()

    training_loop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=EPOCHS,
        label_encoder=label_encoder,
        target_feature=config["target_feature"],
        wandb_run=wandb_run,
        validation_name="ISAdetect_validation",
    )
    
    # try save model if onnx is installed
    if config["validator"].get("save_model", False):
        print("Saving model as ONNX")
        save_model_as_onnx(model=model, model_name=config["model"]["name"], sample_dataset=ISAdetectDataset)

    wandb_run.finish()

    # ======== ISADETECT TESTING ========
    print("\n===== Testing on ISAdetect =====")
    wandb_run = wandb.init(
        project=wandb_project,
        name="ISAdetect_test",
        group=wandb_group_name,
        config=config,
        mode="online" if config["validator"]["wandb_project_name"] else "disabled",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    (
        ISAdetect_testing_accuracies,
        ISAdetect_avg_testing_loss,
        ISAdetect_testing_total_accuracy,
    ) = test_loop(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion,
        label_encoder=label_encoder,
        target_feature=config["target_feature"],
        test_name="ISAdetect",
    )


    wandb_run.log({
        "ISAdetect_testing_accuracy_per_group": wandb.Table(
            data=[[group, acc] for group, acc in ISAdetect_testing_accuracies.items()],
            columns=["group", "accuracy"],
        ),
        "ISAdetect_avg_testing_loss": ISAdetect_avg_testing_loss,
        "ISAdetect_testing_total_accuracy": ISAdetect_testing_total_accuracy,
    })
    print(f"Avg. Test Loss: {ISAdetect_avg_testing_loss:.4f}")
    print(f"Test Total Accuracy: {100*ISAdetect_testing_total_accuracy:.2f}%")

    wandb_run.finish()

    # ======= CPU REC TESTING =======
    print("\n===== Testing on CPU REC =====")
    wandb_run = wandb.init(
        project=wandb_project,
        name="CPU_REC_test",
        group=wandb_group_name,
        config=config,
        mode="online" if config["validator"]["wandb_project_name"] else "disabled",
    )

    cpu_rec_loader = DataLoader(
        CpuRecDataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    (
        cpu_rec_testing_accuracies,
        cpu_rec_avg_testing_loss,
        cpu_rec_testing_total_accuracy,
    ) = test_loop(
        model=model,
        device=device,
        test_loader=cpu_rec_loader,
        criterion=criterion,
        label_encoder=label_encoder,
        target_feature=config["target_feature"],
        test_name="cpu_rec",
    )


    wandb_run.log({
        "cpu_rec_accuracy_per_group": wandb.Table(
            data=[[group, acc] for group, acc in cpu_rec_testing_accuracies.items()],
            columns=["group", "accuracy"],
        ),
        "cpu_rec_avg_testing_loss": cpu_rec_avg_testing_loss,
        "cpu_rec_testing_total_accuracy": cpu_rec_testing_total_accuracy,
    })
    print(f"Avg. Test Loss: {cpu_rec_avg_testing_loss:.4f}")
    print(f"Test Total Accuracy: {100*cpu_rec_testing_total_accuracy:.2f}%")

    wandb_run.finish()

