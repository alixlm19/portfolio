---
title: "Image Classification with Noisy Labels: A Practical Guide"
date: "2024-07-10"
description: "Explore weakly supervised learning by building an image classifier that handles imperfect training labels. Learn techniques to deal with label noise, a common challenge in real-world machine learning projects."
tags: ["Machine Learning", "Computer Vision", "CNN", "Python", "TensorFlow"]
image: ""
featured: true
category: "Machine Learning"
---

In an ideal world, every machine learning project would have perfectly labeled training data. But reality is messier. Getting accurate ground truth labels is often costly, time-consuming, and sometimes impossible. This is where weakly supervised learning comes in—a practical approach to building models when your training labels aren't perfect.

In this article, I'll walk you through a real-world problem: training an image classifier when some of your labels are incorrect. We'll implement techniques from the paper "Multi-Label Fashion Image Classification with Minimal Human Supervision" by Inoue et al., applied to the CIFAR-10 dataset.

## The Challenge of Noisy Labels

Label noise is everywhere in real-world machine learning:

- **Crowdsourced annotations** often contain errors from untrained annotators
- **Automated labeling** systems make mistakes
- **Historical data** may have outdated or incorrect labels
- **Ambiguous examples** can confuse even expert labelers

Traditional supervised learning assumes your training labels are correct. When they're not, your model learns from these mistakes, leading to poor performance on clean test data.

## Understanding the Problem

We're tackling a classification problem where training labels contain noise. This is particularly relevant for image classification tasks where:

1. Some images are inherently ambiguous (is this a dog or a wolf?)
2. Multiple valid labels might exist (an image showing both a car and a person)
3. Annotation errors creep in during large-scale labeling projects

Our goal is to build a model that's robust to these imperfections and still achieves high accuracy on correctly labeled test data.

## The CIFAR-10 Dataset

CIFAR-10 is a classic computer vision benchmark containing 60,000 32×32 color images across 10 classes:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

For this project, we'll intentionally corrupt some of the training labels to simulate real-world label noise, then train our model to be robust against these errors.

## Building a Robust CNN Architecture

Let's start with a convolutional neural network designed to handle noisy training data:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"Training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")
print(f"Image shape: {x_train.shape[1:]}")
```

### Introducing Label Noise

To simulate real-world conditions, we'll randomly flip a percentage of training labels:

```python
def add_label_noise(labels, noise_ratio=0.2, num_classes=10):
    """
    Randomly flip labels to incorrect classes.
    
    Args:
        labels: Original labels
        noise_ratio: Proportion of labels to corrupt (0.0 to 1.0)
        num_classes: Number of classes
    
    Returns:
        Noisy labels
    """
    noisy_labels = labels.copy()
    num_samples = len(labels)
    num_noisy = int(noise_ratio * num_samples)
    
    # Randomly select samples to corrupt
    noisy_indices = np.random.choice(num_samples, num_noisy, replace=False)
    
    for idx in noisy_indices:
        original_class = labels[idx]
        # Choose a different random class
        wrong_classes = list(range(num_classes))
        wrong_classes.remove(original_class[0])
        noisy_labels[idx] = np.random.choice(wrong_classes)
    
    return noisy_labels

# Create noisy training labels (20% corruption)
y_train_noisy = add_label_noise(y_train, noise_ratio=0.2)

print(f"Label accuracy: {np.mean(y_train == y_train_noisy) * 100:.2f}%")
```

### Model Architecture with Noise Robustness

We'll build a CNN with several key features for handling noisy labels:

1. **Batch normalization** to stabilize training
2. **Dropout** to prevent overfitting to noise
3. **Data augmentation** to improve generalization

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation to improve robustness
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(x_train)

def build_robust_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Build a CNN architecture robust to label noise.
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

model = build_robust_cnn()
model.summary()
```

## Training Strategy for Noisy Labels

### Loss Function Selection

Standard cross-entropy loss can be sensitive to label noise. We have several options:

1. **Standard Cross-Entropy** - Simple but can overfit to noisy labels
2. **Label Smoothing** - Softens the target distribution
3. **Symmetric Cross-Entropy** - Explicitly models label noise

Let's implement label smoothing, which prevents the model from becoming overconfident on potentially incorrect labels:

```python
def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    """
    Apply label smoothing to reduce overconfidence.
    
    Instead of [0, 1, 0, 0], targets become [0.025, 0.925, 0.025, 0.025]
    """
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
    y_true = y_true * (1 - smoothing) + smoothing / tf.cast(num_classes, tf.float32)
    
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

# Compile model with label smoothing
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=lambda y_true, y_pred: label_smoothing_loss(y_true, y_pred, smoothing=0.1),
    metrics=['accuracy']
)
```

### Training with Early Stopping

We'll monitor validation accuracy and stop training if the model starts overfitting:

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    # Stop if validation accuracy doesn't improve
    EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True
    ),
    
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    
    # Save best model
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train_noisy, batch_size=128),
    validation_data=(x_test, y_test),
    epochs=100,
    callbacks=callbacks,
    verbose=1
)
```

## Model Evaluation and Analysis

Let's evaluate our model's performance:

```python
# Evaluate on clean test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy on Clean Labels: {test_accuracy * 100:.2f}%")

# Evaluate on noisy training data
train_loss, train_accuracy = model.evaluate(x_train, y_train_noisy)
print(f"Training Accuracy on Noisy Labels: {train_accuracy * 100:.2f}%")

# Compare with accuracy on clean training labels
clean_train_loss, clean_train_accuracy = model.evaluate(x_train, y_train)
print(f"Training Accuracy on Clean Labels: {clean_train_accuracy * 100:.2f}%")
```

### Visualizing Predictions

Let's visualize some predictions to understand where the model succeeds and fails:

```python
import matplotlib.pyplot as plt

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def plot_predictions(images, true_labels, noisy_labels, predictions, num_samples=10):
    """
    Visualize model predictions compared to true and noisy labels.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        
        true_class = class_names[true_labels[i][0]]
        noisy_class = class_names[noisy_labels[i][0]]
        pred_class = class_names[predictions[i]]
        
        # Color code: green if correct, red if wrong
        color = 'green' if predictions[i] == true_labels[i][0] else 'red'
        
        title = f"True: {true_class}\nNoisy: {noisy_class}\nPred: {pred_class}"
        axes[i].set_title(title, color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

# Make predictions
predictions = np.argmax(model.predict(x_test[:10]), axis=1)
plot_predictions(x_test[:10], y_test[:10], y_train_noisy[:10], predictions)
```

### Analyzing Learning Curves

Understanding how the model learned despite noisy labels:

```python
def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_training_history(history)
```

## Key Insights and Best Practices

Through this project, I learned several crucial lessons about handling noisy labels:

### 1. Regularization is Essential

Dropout and batch normalization prevent the model from memorizing noisy labels. Without proper regularization, the model will overfit to the noise.

### 2. Label Smoothing Helps

By softening the target distributions, label smoothing prevents overconfidence on potentially mislabeled examples. This simple technique consistently improves robustness.

### 3. Data Augmentation Matters

Strong data augmentation forces the model to learn more robust features rather than memorizing training examples—including the noisy ones.

### 4. Monitor Multiple Metrics

Don't just watch training accuracy. Compare performance on clean test data vs. noisy training data to understand how well your model generalizes.

### 5. Clean Validation Data is Crucial

Always use clean, correctly labeled data for validation. This ensures you're selecting models based on true performance, not their ability to fit noise.

## Advanced Techniques to Explore

If you want to take this further, consider these advanced approaches:

### Confidence-Based Sample Weighting

Weight training samples based on model confidence. High-confidence predictions are more likely to be correct:

```python
def confidence_based_weighting(y_true, y_pred):
    """
    Weight samples by prediction confidence.
    More confident predictions get higher weight.
    """
    confidence = tf.reduce_max(y_pred, axis=-1)
    weights = tf.pow(confidence, 2)  # Square for emphasis
    return weights
```

### Co-Teaching

Train two networks simultaneously. Each network selects clean samples for the other to learn from:

```python
# Train two models that teach each other
model_1 = build_robust_cnn()
model_2 = build_robust_cnn()

# In each epoch:
# 1. Model 1 selects clean samples for Model 2
# 2. Model 2 selects clean samples for Model 1
# 3. Both models train on selected samples
```

### Label Correction

Attempt to automatically correct noisy labels using model predictions:

```python
def correct_labels(model, x_train, y_train_noisy, confidence_threshold=0.9):
    """
    Correct labels where model is highly confident.
    """
    predictions = model.predict(x_train)
    max_confidence = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    corrected_labels = y_train_noisy.copy()
    high_confidence_mask = max_confidence > confidence_threshold
    
    corrected_labels[high_confidence_mask] = predicted_classes[high_confidence_mask].reshape(-1, 1)
    
    return corrected_labels
```

## Real-World Applications

The techniques we've covered apply to many real-world scenarios:

- **Medical imaging** where expert annotations are expensive and sometimes inconsistent
- **Autonomous driving** where automated labeling systems make mistakes
- **Content moderation** where crowdsourced labels contain errors
- **Historical datasets** with outdated or incorrect labels
- **Multi-annotator datasets** where different experts disagree

## Conclusion

Building machine learning models with imperfect data is a reality we all face. Rather than spending infinite time and money on perfect labels, we can use techniques like:

- Robust architectures with strong regularization
- Label smoothing to prevent overconfidence
- Data augmentation for better generalization
- Smart loss functions that account for noise
- Clean validation data to guide model selection

The key insight is that noisy labels are a feature, not a bug, of real-world machine learning. By designing our systems to handle this noise, we build more practical and deployable solutions.

Remember: perfect data is the enemy of deployed models. Start with what you have, apply robust techniques, and iterate based on real performance.

The complete code for this project is available on my GitHub. Try experimenting with different noise ratios, architectures, and datasets to see how these techniques perform in various scenarios!
