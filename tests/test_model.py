import torch
import pytest
from model import Net
from data_loader import MNISTDataLoader

def test_model_output_shape():
    model = Net()
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), "Model output shape is incorrect"

def test_data_loader():
    loader = MNISTDataLoader(batch_size=32)
    train_loader, test_loader = loader.get_data_loaders()
    assert len(train_loader.dataset) == 60000, "Training dataset size is incorrect"
    assert len(test_loader.dataset) == 10000, "Test dataset size is incorrect" 