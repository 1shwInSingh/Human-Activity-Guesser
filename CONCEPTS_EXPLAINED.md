# ANN HAR Project - Concepts Explained Simply

Here is an easy way to explain your project.

## 1. What the project does

It builds a simple **Artificial Neural Network (ANN)** that guesses a person’s activity from smartphone sensor data.

Example:

> Given sensor readings from a phone, can the model say whether the person is walking, sitting, standing, etc.?

---

## 2. Dataset

You used the **UCI HAR dataset**.

- 30 people carried phones.
- The phone had sensors like accelerometer and gyroscope.
- Each activity window was converted into **561 numeric features**.
- There are **6 classes**:

```text
Walking
Walking Upstairs
Walking Downstairs
Sitting
Standing
Laying
```

Simple explanation:

> The dataset is like 561 clues about how the phone moved. The ANN learns which clue pattern usually belongs to which activity.

---

## 3. Features and labels

A **feature** is an input value.

Example for this project:

- A feature could be a sensor-based number like average acceleration.

A **label** is the correct answer.

```text
Input features: 561 numbers
Label: Walking / Sitting / Standing / ...
```

Simple example:

> In fruit classification, features could be size, color, weight. The label would be apple/banana/mango. In your project, features are sensor values and the label is activity.

---

## 4. Train / validation / test split

Data is divided into three parts:

```text
Training data: model learns from this
Validation data: model is checked while training
Test data: final unseen exam
```

Why?

> Training data is like homework, validation is like weekly tests, and test data is like the final exam.

Your notebook uses a special validation split called **subject-wise split**.

---

## 5. Subject-wise splitting and data leakage

Each row comes from a person.

If the same person appears in both training and validation, the model may “recognize” that person instead of truly learning activity.

That problem is called **data leakage**.

Your project avoids it by putting different people in train and validation.

Simple example:

> It is like training a student only on Teacher A’s question style, but testing them on Teacher A again. That is not fair. Better to train on Teacher A and test on Teacher B.

In your project:

```text
Train people ≠ Validation people
```

---

## 6. Scaling / StandardScaler

Some values may be big and some small.

Neural networks learn better when inputs are in a similar range.

`StandardScaler` converts values so that:

```text
mean ≈ 0
standard deviation ≈ 1
```

It was fitted only on training data, then used on validation/test.

Simple example:

> Imagine exam marks out of 100 and cricket scores out of 300. To compare fairly, we adjust both to the same scale.

---

## 7. Artificial Neural Network

An ANN is inspired by the human brain’s neurons.

It has layers:

```text
Input layer
Hidden layers
Output layer
```

Your architecture:

```text
Input: 561 features
Dense 256 + ReLU + Dropout
Dense 128 + ReLU + Dropout
Output Dense 6 + Softmax
```

Simple explanation:

> The input layer receives clues. Hidden layers combine clues and find patterns. The output layer gives final probabilities.

---

## 8. Neuron, weights, and biases

Each neuron does:

```text
score = input × weight + bias
```

Then it passes through an activation function.

- **Weight**: how important an input is.
- **Bias**: the neuron’s starting offset.
- **Training**: the model adjusts weights and biases to reduce mistakes.

Simple example:

> A neuron deciding “Walking” may give more weight to features related to movement, and less weight to sitting/standing clues.

---

## 9. Dense layer

A dense layer means every neuron connects to every previous input.

Example:

```text
561 input features → 256 neurons
```

So every one of the 256 neurons looks at all 561 features.

---

## 10. ReLU activation

ReLU stands for Rectified Linear Unit.

It does:

```text
if value < 0: 0
else: value
```

Simple explanation:

> It removes useless negative signals and keeps useful positive pattern strength.

Without activation functions, many dense layers would just behave like one layer. ReLU adds non-linearity.

---

## 11. Dropout

Dropout randomly turns off some neurons during training.

Your model used:

```python
keras.layers.Dropout(0.3)
```

This means during training, about 30% of neurons are ignored temporarily.

Why?

> It stops the model from memorizing training data too much.

Simple example:

> Like studying with some books closed at random so you do not depend on one page too much.

---

## 12. L2 regularization

L2 regularization adds a small penalty for very large weights.

Your model used:

```python
kernel_regularizer=keras.regularizers.l2(0.001)
```

Why?

> Very large weights can make the model sensitive to small changes and cause overfitting. L2 encourages simpler weights.

Simple explanation:

> It is like telling the model: “Do not make any rule too extreme.”

---

## 13. Output layer and softmax

The final layer has 6 neurons, one per activity.

Softmax converts raw scores into probabilities.

Example output:

```text
Walking: 0.91
Walking Up: 0.03
Walking Down: 0.01
Sitting: 0.02
Standing: 0.02
Laying: 0.01
```

The model predicts the largest probability.

---

## 14. Sparse categorical crossentropy

This is the loss function.

Loss means:

> How wrong was the prediction?

Sparse categorical crossentropy is used when:

- There are multiple classes.
- Labels are integers like `0, 1, 2, 3, 4, 5`.
- Final layer uses softmax.

If labels were one-hot encoded, you would use categorical crossentropy.

---

## 15. Optimizer Adam

Optimizer is the algorithm that adjusts weights and biases.

Adam is a popular optimizer for neural networks.

It adjusts each weight using:

```text
weight = weight - learning_rate × correction
```

Simple explanation:

> Adam is like the trainer that keeps adjusting every neuron until the model stops making many mistakes.

Learning rate used: `0.001`.

---

## 16. Epochs and batch size

An **epoch** is one full pass through the training data.

Your model trained up to:

```python
epochs=100
batch_size=64
```

But early stopping may stop it sooner.

Simple example:

> One epoch is reading the whole textbook once. Batch size 64 means studying 64 examples at a time before updating the answer.

---

## 17. Early stopping

Early stopping watches validation loss.

Code:

```python
EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
```

Meaning:

- If validation loss does not improve for 10 epochs, stop training.
- Restore the best weights from earlier.

Why?

> If extra training only makes the model memorize more, stop before it gets worse.

---

## 18. Overfitting vs underfitting

Overfitting:

> Model performs very well on training data but poorly on new data.

Underfitting:

> Model is too simple and performs poorly even on training data.

Your project reduces overfitting using:

- Dropout
- L2 regularization
- Early stopping
- Subject-wise split

---

## 19. Accuracy

Accuracy is:

```text
correct predictions / total predictions
```

Your test accuracy:

```text
95.39%
```

Simple explanation:

> Out of about 100 test examples, the model predicted approximately 95 correctly.

---

## 20. Confusion matrix

A confusion matrix shows which classes were confused with which.

Example row:

```text
Actual Sitting, predicted Standing: 42
```

This means some sitting examples were mistaken as standing, which makes sense because they are similar.

For your presentation, say:

> The confusion matrix tells us where the model is strong and which activities it confuses.

---

## 21. Classification report

It gives:

- Precision
- Recall
- F1-score
- Support

Simple meanings:

- Precision: when the model says “Walking”, how often is it correct?
- Recall: out of all actual “Walking” cases, how many did it find?
- F1-score: balance between precision and recall.
- Support: number of examples in that class.

---

## 22. Softmax output and argmax

The model first gives probabilities:

```text
[0.91, 0.03, 0.01, 0.02, 0.02, 0.01]
```

Then `argmax` picks the highest index:

```text
argmax → index 0
```

Index 0 corresponds to:

```text
Walking
```

---

## 23. Model saving and loading

The model was saved as:

```text
har_model.keras
```

The scaler was saved as:

```text
har_scaler.joblib
```

Why save both?

> The model needs weights, but new input must be scaled using the same scaler learned during training.

---

## 24. Frontend webpage

The Streamlit webpage is just a demo.

It lets the user:

- Pick a saved sample
- See the actual activity
- See the ANN prediction
- See confidence percentages
- Show raw sensor features if needed

It does not train the model. It only loads the trained model and predicts.

---

## One-minute teacher explanation script

> We built an ANN project for human activity recognition using the UCI HAR dataset. The dataset contains 561 sensor-based features and 6 activity classes. We split the data by subject to avoid leakage, scaled features using StandardScaler, and trained a dense neural network with two hidden layers, ReLU activation, dropout, L2 regularization, Adam optimizer, and early stopping. The final softmax layer predicts one of six activities. The model reached around 95.39% test accuracy. We also built a Streamlit frontend to demonstrate predictions on saved test samples.

---

## Why this is a good ANN project

Because it clearly shows:

- What the model predicts
- What input it uses
- How an ANN is built
- How weights and biases are learned
- How overfitting is controlled
- How performance is evaluated
- How the trained model can be used in a simple frontend
