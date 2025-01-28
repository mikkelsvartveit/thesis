import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut


class CrossValidator:
    def __init__(
        self,
        model_class,
        criterion,
        optimizer_class,
        dataset,
        batch_size,
        num_epochs,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    ):
        self.model_class: nn.Module = model_class
        self.criterion = criterion
        self.optimizer_class: torch.optim.Optimizer = optimizer_class
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.device: str = device

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, model: nn.Module, val_loader: DataLoader):
        model.eval()
        total_loss = 0
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        return total_loss / len(val_loader), np.array(predictions), np.array(actuals)

    def k_fold_cross_validation(self, k=5):
        kfold = KFold(n_splits=k, shuffle=True)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            print(f"Fold {fold + 1}/{k}")

            # Create data loaders for this fold
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=train_sampler
            )
            val_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=val_sampler
            )

            # Initialize model and optimizer
            model = self.model_class().to(self.device)
            optimizer = self.optimizer_class(model.parameters())

            # Training loop
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch(model, train_loader, optimizer)
                val_loss, predictions, actuals = self.validate(model, val_loader)
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )

            fold_results.append(
                {"val_loss": val_loss, "predictions": predictions, "actuals": actuals}
            )

        return fold_results

    def leave_one_out_cv(self):
        loo = LeaveOneOut()
        results = []

        for train_idx, val_idx in loo.split(self.dataset):
            # Create data loaders
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=train_sampler
            )
            val_loader = DataLoader(self.dataset, batch_size=1, sampler=val_sampler)

            # Initialize model and optimizer
            model = self.model_class().to(self.device)
            optimizer = self.optimizer_class(model.parameters())

            # Training loop
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch(model, train_loader, optimizer)

            # Validation on held-out sample
            val_loss, predictions, actuals = self.validate(model, val_loader)
            results.append(
                {"val_loss": val_loss, "predictions": predictions, "actuals": actuals}
            )

        return results

    def leave_one_group_out_cv(self, groups):
        """
        Performs Leave-One-Group-Out cross-validation

        Args:
            groups: array-like of shape (n_samples,)
                Group labels for the samples used while splitting the dataset

        Returns:
            List of dictionaries containing validation results for each group
        """
        logo = LeaveOneGroupOut()
        results = []

        for fold, (train_idx, val_idx) in enumerate(
            logo.split(self.dataset, groups=groups)
        ):
            print(f"Fold {fold + 1}/{len(groups)}")
            print(f"Validating on group {groups[val_idx[0]]}")

            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            train_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=train_sampler
            )
            val_loader = DataLoader(
                self.dataset, batch_size=self.batch_size, sampler=val_sampler
            )

            model = self.model_class().to(self.device)
            optimizer = self.optimizer_class(model.parameters())

            # Training loop
            for epoch in range(self.num_epochs):
                train_loss = self.train_epoch(model, train_loader, optimizer)
                val_loss, predictions, actuals = self.validate(model, val_loader)
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} - "
                    f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )

            results.append(
                {
                    "group": groups[val_idx[0]],
                    "val_loss": val_loss,
                    "predictions": predictions,
                    "actuals": actuals,
                }
            )

        return results
