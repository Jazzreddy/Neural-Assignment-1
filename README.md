# Neural-Assignment-1
# Neural Networks and Deep Learning - Assignment

## Student Information
Name:
Student ID:

---

## Overview
This repository contains my implementation of the first home assignment for the Neural Networks and Deep Learning course. The tasks involve tensor manipulations, loss function comparisons, training models with different optimizers, and TensorBoard logging.

---

## Task 1: Tensor Manipulations & Reshaping
### Steps:
1. *Created a random tensor* of shape (4,6) using TensorFlow.
2. *Found its rank and shape* before reshaping.
3. *Reshaped it into (2,3,4)* and transposed it to (3,2,4).
4. *Performed broadcasting* of a smaller tensor (1,4) to match the larger tensor.
5. *Added the tensors* to demonstrate broadcasting.

### Key Concept: Broadcasting
Broadcasting allows smaller tensors to be automatically expanded to match the dimensions of a larger tensor during arithmetic operations.

---

## Task 2: Loss Functions & Hyperparameter Tuning
### Steps:
1. *Defined true values (y_true)* and model predictions (y_pred).
2. *Computed two loss functions:*
   - Mean Squared Error (MSE)
   - Categorical Cross-Entropy (CCE)
3. *Modified predictions slightly* and checked how loss values change.
4. *Plotted a bar chart* comparing MSE and CCE loss.

### Observations:
- MSE is more sensitive to small errors in predictions.
- CCE is useful for classification tasks with probability outputs.

---

## Task 3: Training a Model with Different Optimizers
### Steps:
1. *Loaded the MNIST dataset* (handwritten digit images).
2. *Trained two models:*
   - One with Adam optimizer.
   - One with SGD optimizer.
3. *Compared training and validation accuracy trends.*
4. *Plotted accuracy curves* for Adam vs. SGD.

### Observations:
- *Adam optimizer* generally converges faster.
- *SGD optimizer* has a slower but steadier learning process.

---

## Task 4: Training a Neural Network and Logging to TensorBoard
### Steps:
1. *Loaded and preprocessed the MNIST dataset.*
2. *Trained a simple neural network model* with TensorBoard logging enabled.
3. *Stored logs in logs/fit/ directory.*
4. *Launched TensorBoard* to analyze accuracy and loss trends.

### Questions & Answers:
1. *Patterns observed in training vs. validation accuracy:*
   - If the validation accuracy is much lower than training accuracy, the model might be overfitting.
2. *Using TensorBoard to detect overfitting:*
   - Look for a widening gap between training and validation loss.
3. *Effect of increasing epochs:*
   - Training accuracy improves, but validation accuracy may plateau or drop due to overfitting.

---

## Running the Code
To run this project, follow these steps:
1. Clone the repository:
   bash
   git clone [Your Repo URL]
   cd [Your Repo Folder]
   
2. Install dependencies:
   bash
   pip install tensorflow numpy matplotlib
   
3. Run the script:
   bash
   python neural_net_assignment.py
   
4. View TensorBoard logs:
   bash
   tensorboard --logdir=logs/fit
   

---

## Video Submission
A 2-3 minute video has been created to:
- Demonstrate tensor manipulations.
- Explain loss function behavior.
- Compare optimizer performance.
- Show TensorBoard logging and analysis.

---

## Conclusion
This assignment provided hands-on experience with TensorFlow operations, loss function evaluation, model training, and TensorBoard visualization. These foundational concepts are essential for developing deep learning models effectively.

---

## References
- TensorFlow Documentation: https://www.tensorflow.org/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
