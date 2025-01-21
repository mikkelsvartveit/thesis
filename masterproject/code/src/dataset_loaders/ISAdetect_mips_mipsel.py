import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np

class BinaryFileDataset(Dataset):
    def __init__(self, mips_dir, mipsel_dir, transform=None):
        self.transform = transform
        self.files = []
        self.labels = []
        
        # Collect MIPS files (label 0)
        mips_files = Path(mips_dir).glob('*.code')
        for file_path in mips_files:
            self.files.append(file_path)
            self.labels.append(0)  # 0 for MIPS
            
        # Collect MIPSEL files (label 1)
        mipsel_files = Path(mipsel_dir).glob('*.code')
        for file_path in mipsel_files:
            self.files.append(file_path)
            self.labels.append(1)  # 1 for MIPSEL
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        
        # Read binary file
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        
        # Convert to numpy array for easier processing
        data = np.frombuffer(binary_data, dtype=np.uint8)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(data)
        else:
            features = torch.from_numpy(data)
        
        return features, label

class BytePatternTransform:
    def __init__(self):
        self.patterns = [
            b'\x00\x01',  # 0x0001
            b'\x01\x00',  # 0x0100
            b'\xfe\xff',  # 0x0100
            b'\xff\xfe'   # 0x1011
        ]
    
    def __call__(self, data):
        # Convert data to bytes if it's not already
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        
        # Count occurrences of each pattern
        counts = []
        for pattern in self.patterns:
            count = data.count(pattern)
            counts.append(count)
        
        return torch.tensor(counts, dtype=torch.float32)

# Usage example:
def create_train_test_dataloaders(
    mips_dir, 
    mipsel_dir, 
    batch_size=32, 
    num_workers=4, 
    test_split=0.2,
    random_seed=42
):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Create full dataset
    transform = BytePatternTransform()
    full_dataset = BinaryFileDataset(
        mips_dir=mips_dir,
        mipsel_dir=mipsel_dir,
        transform=transform
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

if __name__ == "__main__":
    # Replace with your actual paths
    mips_dir = "dataset/ISAdetect/ISA_detect_full_dataset/mips"
    mipsel_dir = "dataset/ISAdetect/ISA_detect_full_dataset/mipsel"
    
    # Create train and test loaders with 80-20 split
    train_loader, test_loader = create_train_test_dataloaders(
        mips_dir=mips_dir,
        mipsel_dir=mipsel_dir,
        test_split=0.2,
    )
    
    # Print dataset sizes
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test the loaders
    print("\nSample from training loader:")
    for batch_features, batch_labels in train_loader:
        print("Feature shape:", batch_features.shape)
        print("Labels shape:", batch_labels.shape)
        print("Sample features:", batch_features[0])
        print("Sample label:", batch_labels[0])
        break
        
    print("\nSample from test loader:")
    for batch_features, batch_labels in test_loader:
        print("Feature shape:", batch_features.shape)
        print("Labels shape:", batch_labels.shape)
        print("Sample features:", batch_features[0])
        print("Sample label:", batch_labels[0])
        break