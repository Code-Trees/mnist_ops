import torch
from model import Net
from optimizer import get_optimizer,run_lrfinder
from model_fit import training,testing
import albumentations as A
from albumentations.augmentations.geometric.resize import Resize
from albumentations.pytorch.transforms import ToTensorV2
from data_loader import MNISTDataLoader
import json
from torch import nn
import plotext as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_model(device):
    model = Net().to(device)
    return model


def run_model(model,device,batch_size,epochs,optimizer,scheduler,best_model):
    train_losses = []
    train_accuracy = []
    test_losses =[]
    test_accuracy = []
    for EPOCHS in range(0,epochs):
        train_loss, train_acc = training(model,device,train_loader,optimizer,EPOCHS)
        train_losses.append(train_loss)
        train_accuracy.append(train_acc)

        test_loss,test_acc = testing(model,device,test_loader,EPOCHS)
        test_accuracy.append(test_acc)
        test_losses.append(test_loss)
        
        scheduler.step()
        try:
            if len(test_accuracy) > 1:
                if (EPOCHS >= 3 and 
                    max(test_accuracy[:-1]) < test_accuracy[-1] and 
                    max(test_accuracy) >= best_model):
                    
                    checkpoint = {
                        'epoch': EPOCHS + 1,
                        'valid_loss_min': test_losses[-1],
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    
                    file_name = f"./model_folder/modelbest_{test_accuracy[-1]:.4f}_epoch_{EPOCHS}.pt"
                    torch.save(checkpoint, file_name)
                    print(f"Target Achieved: {max(test_accuracy) * 100:.2f}% Test Accuracy!!")
                else:
                    print("Conditions not met for saving the model.")
            else:
                print("Insufficient test accuracy data.")
        except Exception as e:
            print(f"Model saving failed: {e}")

        print(f"LR: {scheduler.get_lr()[0]}\n")
    return model,train_losses, train_accuracy,test_losses,test_accuracy


def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)


def get_loss_function(loss_type):
    if loss_type is None:
        return nn.CrossEntropyLoss()   
    loss_types = {
        'cross_entropy': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
        'nll': nn.NLLLoss()
    }
    return loss_types.get(loss_type.lower(), nn.CrossEntropyLoss())


if __name__ == "__main__":
    config = load_config()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Get loss function and scheduler settings from config
    loss_fn = get_loss_function(config['training'].get('loss_type'))
    use_scheduler = config['training'].get('use_scheduler', False)
    scheduler_type = config['training'].get('scheduler_type', 'steplr')
    use_scheduler = bool(use_scheduler)
    best_model = config['best_model']

    # Set seed from config
    _ = torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(config['seed'])
    
    # Get batch size based on device
    batch_size = config['training']['batch_size']
    epochs = config['training']['epochs']
    
    model = get_model(device)
    data_loader = MNISTDataLoader(batch_size=batch_size)
    train_loader, test_loader = data_loader.get_data_loaders()

    
    lrs,_ = run_lrfinder(
        model, 
        device, 
        train_loader, 
        test_loader, 
        start_lr=config['training']['start_lr'],
        end_lr=config['training']['end_lr']
    )
    optimizer,scheduler = get_optimizer(model,scheduler = use_scheduler,\
                              scheduler_type = scheduler_type,lr = lrs[0])

    model,train_losses, train_accuracy,test_losses,test_accuracy= run_model(model,device,batch_size,epochs,optimizer,scheduler,best_model)

    # Set a canvas size suitable for your terminal
    plt.plotsize(5,5)

    # Plot for accuracy
    plt.clf()  # Clear the canvas before starting a new plot
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(test_accuracy, label="Test Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    
    plt.plotsize(5,5)
    # Plot for loss
    plt.clf()  # Clear the canvas again for the next plot
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

