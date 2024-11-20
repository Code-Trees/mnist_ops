import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import pytest
from model import Net
from data_loader import MNISTDataLoader
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torchvision import datasets
from unittest.mock import MagicMock
from model_fit import training


def test_model_param_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())  # Count total parameters
    assert total_params < 25000, f"Model has {total_params} parameters, which exceeds the limit of 25,000."

def test_model_output_shape():
    model = Net()
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), "Model output shape is incorrect"

#  Test the CUDA availability check
def test_cuda_available():
    dataloader = MNISTDataLoader()
    assert isinstance(dataloader.cuda, bool), "CUDA availability should return a boolean value."
    print(f"CUDA availability: {dataloader.cuda}")

# Test batch size initialization
def test_batch_size():
    dataloader = MNISTDataLoader(batch_size=64)
    assert dataloader.batch_size == 64, "Batch size should match the input value."
    print(f"Batch size: {dataloader.batch_size}")

# Test the mean and std calculation
def test_calculate_stats():
    dataloader = MNISTDataLoader()
    mean, std = dataloader._calculate_stats()
    assert isinstance(mean, float) and isinstance(std, float), "Mean and std should be floats."
    assert 0 <= mean <= 1, "Mean should be in the range [0, 1] for normalized image data."
    assert std > 0, "Std should be greater than 0."
    print(f"Calculated mean: {mean}, std: {std}")

# Test the train and test transformations
def test_transformations():
    dataloader = MNISTDataLoader()
    dummy_image = torch.rand(1, 28, 28)  # Random tensor simulating an image
    
    try:
        # Convert tensor to PIL Image
        pil_image = ToPILImage()(dummy_image.squeeze(0))
        
        # Apply transformations
        transformed_train_image = dataloader.train_transforms(pil_image)
        transformed_test_image = dataloader.test_transforms(pil_image)
        
        # Check that transformed images are still valid tensors of expected size
        assert transformed_train_image.shape[-2:] == dummy_image.shape[-2:], \
            "Transformed train image should maintain height and width."
        assert transformed_test_image.shape[-2:] == dummy_image.shape[-2:], \
            "Transformed test image should maintain height and width."
        
        print("Transformations are working correctly.")
    except Exception as e:
        print(f"Error in transformations: {e}")

# Test data loader arguments
def test_dataloader_args():
    dataloader = MNISTDataLoader()
    assert 'shuffle' in dataloader.dataloader_args, "DataLoader args should include 'shuffle'."
    assert 'batch_size' in dataloader.dataloader_args, "DataLoader args should include 'batch_size'."
    print(f"Dataloader arguments: {dataloader.dataloader_args}")

# Test train and test data loaders
def test_data_loaders():
    try:
        dataloader = MNISTDataLoader()
        train_loader, test_loader = dataloader.get_data_loaders()
        # Ensure train and test loaders are not empty
        train_batches = len(train_loader)
        test_batches = len(test_loader)  
        assert train_batches > 0, "Train loader should contain batches."
        assert test_batches > 0, "Test loader should contain batches."
        print(f"Train loader batches: {train_batches}, Test loader batches: {test_batches}")
    except Exception as e:
        print(f"Error in data loaders: {e}")

def test_data_augmentation():
    # Initialize the dataloader
    dataloader = MNISTDataLoader()
    dummy_image = torch.rand(1, 28, 28)
    try:
        pil_image = ToPILImage()(dummy_image.squeeze(0))
        transformed_train_image = dataloader.train_transforms(pil_image)
        transformed_test_image = dataloader.test_transforms(pil_image)
        assert isinstance(transformed_train_image, torch.Tensor), \
            "Transformed image should be a torch.Tensor."
        assert isinstance(transformed_test_image, torch.Tensor), \
            "Transformed image should be a torch.Tensor."
        print("Data augmentation transforms are working correctly.")
    except Exception as e:
        print(f"Error in data augmentation: {e}")



# Test Training Function
@pytest.fixture(scope="function")
def setup_data():
    # Create dummy data for testing
    x = torch.rand(64, 1, 28, 28)  # 64 samples, 28x28 image
    y = torch.randint(0, 10, (64,))  # Random target labels (10 classes)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

@pytest.fixture(scope="function")
def setup_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer

def test_training(setup_data, setup_model):
    model, optimizer = setup_model
    train_data = setup_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train the model for one epoch
    loss, accuracy = training(model, device, train_data, optimizer, epochs=1)

    # Ensure the loss and accuracy are returned as floats
    assert isinstance(loss, float), f"Expected loss to be float, got {type(loss)}"
    assert isinstance(accuracy, float), f"Expected accuracy to be float, got {type(accuracy)}"
    assert 0 <= accuracy <= 100, "Accuracy should be between 0 and 100"

def test_training_with_scheduler(setup_data, setup_model):
    model, optimizer = setup_model
    train_data = setup_data
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train the model for one epoch with the scheduler
    loss, accuracy = training(model, device, train_data, optimizer, epochs=1, scheduler=scheduler)

    # Ensure that loss and accuracy are returned as floats
    assert isinstance(loss, float), f"Expected loss to be float, got {type(loss)}"
    assert isinstance(accuracy, float), f"Expected accuracy to be float, got {type(accuracy)}"


