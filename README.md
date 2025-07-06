# MNIST Handwritten Digit Classification Using Convolutional Neural Networks

**Author:** Preyash Shah  
**Roll No:** 24B2184

## Table of Contents

- [Introduction](#introduction)
- [Data Preparation and Augmentation](#data-preparation-and-augmentation)
- [CNN Architecture Design](#cnn-architecture-design)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Analysis and Conclusion](#analysis-and-conclusion)
 
## Introduction

The objective of this project is to **design, train, and evaluate a Convolutional Neural Network (CNN) for the classification of handwritten digits using the MNIST dataset**. This project tests understanding of data preparation, CNN architecture, and the practical application of deep learning frameworks like PyTorch.

The **MNIST dataset** is a cornerstone of machine learning and computer vision. It contains 70,000 grayscale images of handwritten digits (0 through 9), split into a training set of 60,000 images and a testing set of 10,000 images. Each image is a 28×28 pixel square. The aim is to build a model that can look at one of these images and correctly predict the digit it represents. This project mirrors real-world tasks like optical character recognition (OCR) in postal services and banking.

## Data Preparation and Augmentation

The MNIST dataset was loaded using PyTorch’s `torchvision.datasets`. To ensure correct data loading, a batch of images was visualized with their corresponding labels. **Preprocessing** included converting images to tensors and normalizing them using the dataset’s mean (0.1307) and standard deviation (0.3081).

**Normalization** is a crucial preprocessing step when training neural networks because it brings all input pixel values to a standard scale, leading to faster convergence, better stability during training, and improved performance. For MNIST, normalization was applied using:

- **Mean ($$\mu$$)**: 0.1307
- **Standard Deviation ($$\sigma$$)**: 0.3081

These values were calculated empirically by iterating through all images and applying standard formulas.

**Data augmentation techniques** such as random rotation and translation were applied to the training set to artificially increase data diversity and improve model generalization, thereby reducing the risk of overfitting. By seeing different variations of the same digit during training, the model learns to recognize digits regardless of their position, orientation, or scale.

### Augmentation Techniques Used

The following augmentation techniques were applied to the training dataset:

- **RandomRotation(10):** Rotates the image by a random angle in the range $$[-10^\circ, +10^\circ]$$ to simulate slanted handwriting.
- **RandomResizedCrop(28, scale=(0.7, 1.0), ratio=(0.9, 1.1)):** Randomly crops and resizes the image to the original size (28×28), introducing variation in the scale and aspect ratio of digits.
- **RandomAffine(0, translate=(0.1, 0.1)):** Applies random translations (up to 10% of image width and height) to mimic different digit placements within the image.

## CNN Architecture Design

The CNN used in this project is a multi-layer architecture designed to classify 28×28 grayscale handwritten digits from the MNIST dataset. The model comprises two convolutional layers, each followed by ReLU activation and max pooling. The output is then flattened and passed through a fully connected layer with 128 neurons, and finally, an output layer with LogSoftmax activation for 10 classes.

| Layer Type        | Parameters                                        |
|-------------------|--------------------------------------------------|
| Conv2D #1         | 1→32, 3×3 kernel, stride=1, padding=1            |
| ReLU              | Activation function                              |
| MaxPool2D #1      | 2×2 kernel, stride=2                             |
| Conv2D #2         | 32→64, 3×3 kernel, stride=1, padding=1           |
| ReLU              | Activation function                              |
| MaxPool2D #2      | 2×2 kernel, stride=2                             |
| Flatten           | -                                                |
| Fully Connected   | 64×7×7→128                                       |
| Output (FC)       | 128→10, LogSoftmax                               |

**Activation Function:**  
ReLU (Rectified Linear Unit) is applied after each convolutional layer to introduce non-linearity, which helps to learn complex patterns and avoid vanishing gradients.

**Pooling:**  
Pooling reduces the spatial dimensions, preserving important features while reducing computation and overfitting risk. MaxPooling selects the most dominant features by taking the maximum value in each window.

**Output Layer:**  
The final layer applies LogSoftmax activation to output log-probabilities for 10 digit classes. This is suitable for multi-class classification tasks and works with the NLLLoss used during training.

## Model Training and Evaluation

The model was trained using the **Adam optimizer** (learning rate 0.001) and **Negative Log Likelihood Loss (NLLLoss)**. Training was conducted over 10 epochs with a batch size of 64. The training loop included forward passes, loss computation, backpropagation, and weight updates.

After training, the model was evaluated on the test set. The final test accuracy achieved was **98.93%**.

### Loss Function

For this multi-class classification task, **Negative Log Likelihood Loss (NLLLoss)** was selected as the loss function, as it is directly connected to **LogSoftmax** as the activation function in the output layer. LogSoftmax converts the raw network outputs (logits) into log-probabilities for each class. NLLLoss then measures how well the predicted log-probabilities align with the true class labels. This combination is mathematically stable and efficient.

> Using CrossEntropyLoss would only be suitable if the output layer produced raw logits, since CrossEntropyLoss internally applies LogSoftmax before computing the loss.

### Optimizer

The network is trained using the **Adam optimizer** with a learning rate of `0.001`. Adam (Adaptive Moment Estimation) achieves faster and stable convergence compared to SGD by adapting the learning rate for each parameter.

### Training Loop Description

The model is trained over 10 epochs using the standard PyTorch training loop. Each epoch involves:

1. **Model Training Mode:** Set the model to training mode (`model.train()`).
2. **Iterate Over Batches:** For each batch:
   - *Forward Pass*: Pass the batch through the model to obtain predicted log-probabilities.
   - *Loss Calculation*: Compute the loss using NLLLoss.
   - *Zero Gradients*: Reset gradients to zero.
   - *Backward Pass*: Compute gradients via backpropagation.
   - *Parameter Update*: Update model parameters using Adam.
   - *Loss Accumulation*: Accumulate batch loss for average epoch loss.
3. **Repeat:** Continue for all batches in the epoch.

After each epoch, the model is evaluated on the test dataset in evaluation mode (`model.eval()`), and validation loss is computed without gradient updates.

## Analysis and Conclusion

### Model Performance

The trained CNN model achieved a final test accuracy of **98.93%**, which is up to expectations for MNIST using a two-layer convolutional architecture.

![Confusion Matrix confusion matrix above shows that the model occasionally confused digits with similar handwritten forms, particularly:

- **2 and 7:** The bottom line in 2, if minimized, can look like 7.
- **4 and 9:** If the "wick" of 4 touches, it can look like 9.
- **6 and 5:** An incomplete round in 6 can look like 5.

This suggests that while the model performs well overall, more particular features are needed to distinguish similar digit pairs.

### Challenges Faced

- Data augmentation sometimes distorted digits excessively. For example, applying vertical flip caused confusion between 6 and 9, and horizontal flip distorted digits like 5.
- Adding a dropout layer did not significantly affect accuracy.
- Writing this report in LaTeX posed some challenges.
- Ensuring the model did not become biased towards certain digits.

### Potential Improvements

1. Applying a dynamic learning rate can help achieve better convergence and prevent getting stuck in local minima.
2. Adding a third convolutional layer or increasing filter sizes may help the network capture more complex patterns, potentially boosting accuracy on harder-to-classify digits.
3. Incorporating more advanced augmentations without adversely affecting the dataset.
