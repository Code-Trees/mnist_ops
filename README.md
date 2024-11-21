# Model Documentation

## Model Architecture
This deep learning model is a Convolutional Neural Network (CNN) designed for image classification tasks. The architecture is inspired by many people with additional modifications for improved performance.

### Key Features:
- Residual connections for better gradient flow
- Automatic data scaling
- Batch normalization layers for stable training
- Dropout layers for regularization
- Global average pooling to reduce parameters
- Softmax activation for multi-class classification
- Less parameter-based model, best performance
- LR-Finder / Scheduler for faster training
- Modularized code for easy maintenance
- 11 test cases for GitHub action deployment
- CPU and GPU-based training

### Architecture Details:

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
       BatchNorm2d-2           [-1, 10, 26, 26]              20
           Dropout-3           [-1, 10, 26, 26]               0
              ReLU-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]             900
       BatchNorm2d-6           [-1, 10, 24, 24]              20
           Dropout-7           [-1, 10, 24, 24]               0
              ReLU-8           [-1, 10, 24, 24]               0
            Conv2d-9           [-1, 10, 22, 22]             900
      BatchNorm2d-10           [-1, 10, 22, 22]              20
          Dropout-11           [-1, 10, 22, 22]               0
             ReLU-12           [-1, 10, 22, 22]               0
           Conv2d-13           [-1, 10, 20, 20]             900
      BatchNorm2d-14           [-1, 10, 20, 20]              20
          Dropout-15           [-1, 10, 20, 20]               0
             ReLU-16           [-1, 10, 20, 20]               0
        MaxPool2d-17           [-1, 10, 10, 10]               0
           Conv2d-18             [-1, 10, 8, 8]             900
      BatchNorm2d-19             [-1, 10, 8, 8]              20
          Dropout-20             [-1, 10, 8, 8]               0
             ReLU-21             [-1, 10, 8, 8]               0
           Conv2d-22             [-1, 16, 6, 6]           1,440
      BatchNorm2d-23             [-1, 16, 6, 6]              32
          Dropout-24             [-1, 16, 6, 6]               0
             ReLU-25             [-1, 16, 6, 6]               0
           Conv2d-26             [-1, 16, 4, 4]           2,304
      BatchNorm2d-27             [-1, 16, 4, 4]              32
          Dropout-28             [-1, 16, 4, 4]               0
             ReLU-29             [-1, 16, 4, 4]               0
        AvgPool2d-30             [-1, 16, 1, 1]               0
           Conv2d-31             [-1, 10, 1, 1]             160
================================================================
Total params: 7,758
Trainable params: 7,758
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.03
Estimated Total Size (MB): 0.74
----------------------------------------------------------------
```

### Model Parameters
- Total Parameters: 7.7K
- Trainable Parameters: 7.7K
- Input Shape: `(1, 28, 28)`
- Output Classes: 10

## Data Augmentation Pipeline

### Image Augmentation Techniques

1. **Geometric Transformations**
   - Random rotation (±18°): Helps in rotation invariance
   - Scaling with mean and STD

## Model Running Pipeline

### LR Finders

for 100 Iter:

```python
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 64.31it/s]
Learning rate search finished. See the graph with {finder_name}.plot()
LR suggestion: steepest gradient
Suggested LR: 1.56E+00
```

![Alt text](readme_images/download.png)

100 Iter Seems Very less , Lets run it for 1000 Iterations:

```python
 96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████      | 957/1000 [00:15<00:00, 63.40it/s]
Stopping early, the loss has diverged
Learning rate search finished. See the graph with {finder_name}.plot()
LR suggestion: steepest gradient
Suggested LR: 2.74E-02
```

![Alt text](readme_images/download_lr2.png)



The model will automatic pick the Best learning rate . For Small models like MNIST it is not so important. But for bigger model it is going to save real money by reducing your training time 

### Training Epochs

```python
Loss: 0.22103686287496876 LR :0.027380251779278577
Train ==> Epochs: 0 Batch:  483 loss: 0.20563443005084991 Accuracy: 87.63% : 100%|██████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.61it/s]
Test ==> Epochs: 0 Batch:  80 loss: 0.0008179874941706657 Accuracy: 96.77% : 100%|████████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 59.35it/s]
Insufficient test accuracy data.
LR: 0.027380251779278577

Train ==> Epochs: 1 Batch:  483 loss: 0.10929416120052338 Accuracy: 96.58% : 100%|██████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 61.60it/s]
Test ==> Epochs: 1 Batch:  80 loss: 0.0005644781629554927 Accuracy: 97.88% : 100%|████████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 58.30it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 2 Batch:  483 loss: 0.07642707973718643 Accuracy: 97.16% : 100%|██████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 61.36it/s]
Test ==> Epochs: 2 Batch:  80 loss: 0.0005128854883834719 Accuracy: 97.91% : 100%|████████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 57.49it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 3 Batch:  483 loss: 0.037454623728990555 Accuracy: 97.39% : 100%|█████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.87it/s]
Test ==> Epochs: 3 Batch:  80 loss: 0.000509832770191133 Accuracy: 98.01% : 100%|█████████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 56.67it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 4 Batch:  483 loss: 0.1073000356554985 Accuracy: 97.56% : 100%|███████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.33it/s]
Test ==> Epochs: 4 Batch:  80 loss: 0.00044302357528358695 Accuracy: 98.25% : 100%|███████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 57.66it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 5 Batch:  483 loss: 0.015759916976094246 Accuracy: 97.78% : 100%|█████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.36it/s]
Test ==> Epochs: 5 Batch:  80 loss: 0.0004236392221413553 Accuracy: 98.33% : 100%|████████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 58.49it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 6 Batch:  483 loss: 0.052235331386327744 Accuracy: 97.79% : 100%|█████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 61.38it/s]
Test ==> Epochs: 6 Batch:  80 loss: 0.0004259990774095058 Accuracy: 98.20% : 100%|████████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 57.61it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 7 Batch:  483 loss: 0.0445740781724453 Accuracy: 97.80% : 100%|███████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.83it/s]
Test ==> Epochs: 7 Batch:  80 loss: 0.00042445806255564096 Accuracy: 98.29% : 100%|███████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 60.62it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 8 Batch:  483 loss: 0.09609514474868774 Accuracy: 97.91% : 100%|██████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.13it/s]
Test ==> Epochs: 8 Batch:  80 loss: 0.00040606864597648383 Accuracy: 98.36% : 100%|███████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 59.32it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 9 Batch:  483 loss: 0.05668123438954353 Accuracy: 97.94% : 100%|██████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 62.50it/s]
Test ==> Epochs: 9 Batch:  80 loss: 0.00036366956655401735 Accuracy: 98.64% : 100%|███████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 58.46it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 10 Batch:  483 loss: 0.0752599760890007 Accuracy: 98.01% : 100%|██████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 61.65it/s]
Test ==> Epochs: 10 Batch:  80 loss: 0.00038592721642926334 Accuracy: 98.49% : 100%|██████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 57.29it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 11 Batch:  483 loss: 0.03232988342642784 Accuracy: 98.12% : 100%|█████████████████████████████████████████████████████████████| 484/484 [00:08<00:00, 58.79it/s]
Test ==> Epochs: 11 Batch:  80 loss: 0.00038731231093406675 Accuracy: 98.59% : 100%|██████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 56.28it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 12 Batch:  483 loss: 0.11568798124790192 Accuracy: 98.13% : 100%|█████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 61.70it/s]
Test ==> Epochs: 12 Batch:  80 loss: 0.0003816935612820089 Accuracy: 98.60% : 100%|███████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 57.01it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 13 Batch:  483 loss: 0.08745071291923523 Accuracy: 98.07% : 100%|█████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 60.90it/s]
Test ==> Epochs: 13 Batch:  80 loss: 0.00042680786657147107 Accuracy: 98.40% : 100%|██████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 59.17it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Train ==> Epochs: 14 Batch:  483 loss: 0.048578690737485886 Accuracy: 98.15% : 100%|████████████████████████████████████████████████████████████| 484/484 [00:07<00:00, 61.48it/s]
Test ==> Epochs: 14 Batch:  80 loss: 0.00033380933504085987 Accuracy: 98.68% : 100%|██████████████████████████████████████████████████████████████| 81/81 [00:01<00:00, 59.72it/s]
Conditions not met for saving the model.
LR: 0.027380251779278577

Max Train Accuracy:  0.9815
Max Test Accuracy:  0.9868

```

- Loss convergence achieved at   11th epoch 
- Learning rate annealing helped prevent over fitting.

![Alt text](readme_images/Train_test.png)

### Deployment

**Test cases**

Test case help to validate that I am pushing Correct information to  repo. Let's test it locally:

```python
============================= test session starts ==============================
platform linux -- Python 3.12.7, pytest-8.3.3, pluggy-1.5.0 -- /bin/python
cachedir: .pytest_cache
rootdir: Mnist_ops
plugins: anyio-4.6.2.post1
collected 11 items                                                             

tests/test_model.py::test_model_param_count PASSED                       [  9%]
tests/test_model.py::test_model_output_shape PASSED                      [ 18%]
tests/test_model.py::test_cuda_available PASSED                          [ 27%]
tests/test_model.py::test_batch_size PASSED                              [ 36%]
tests/test_model.py::test_calculate_stats PASSED                         [ 45%]
tests/test_model.py::test_transformations PASSED                         [ 54%]
tests/test_model.py::test_dataloader_args PASSED                         [ 63%]
tests/test_model.py::test_data_loaders PASSED                            [ 72%]
tests/test_model.py::test_data_augmentation PASSED                       [ 81%]
tests/test_model.py::test_training PASSED                                [ 90%]
tests/test_model.py::test_training_with_scheduler PASSED                 [100%]

============================= 11 passed in 36.37s ==============================
```



## Push to git hub with Git action configured



![GithubAction](readme_images/Git_log.png)



## Requirements

- Python 3.8+
- PyTorch 1.10+
- Albumentations 1.1.0
- NumPy 1.21+
- OpenCV 4.5+
- CUDA 11.3+ (for GPU support)

