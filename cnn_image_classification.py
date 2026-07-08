"""
CNN Image Classification Project — CIFAR-10
=============================================
A complete, beginner-friendly pipeline for classifying images using a
Convolutional Neural Network (CNN) with TensorFlow/Keras.

Dataset: CIFAR-10 (60,000 32x32 color images, 10 classes)
Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Run this script top to bottom, or copy sections into a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. LOAD & EXPLORE THE DATA
# ---------------------------------------------------------------------------
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Training set: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test set:     {x_test.shape}, Labels: {y_test.shape}")

# Visualize a few sample images
def plot_sample_images(x, y, class_names, n=10):
    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        plt.title(class_names[y[i][0]])
        plt.axis('off')
    plt.suptitle("Sample Training Images")
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=100)
    plt.close()
    print("Saved sample_images.png")

plot_sample_images(x_train, y_train, class_names)

# Check class balance
unique, counts = np.unique(y_train, return_counts=True)
print("\nClass distribution (training set):")
for cls, cnt in zip(unique, counts):
    print(f"  {class_names[cls]}: {cnt}")

# ---------------------------------------------------------------------------
# 2. PREPROCESS THE DATA
# ---------------------------------------------------------------------------
# Normalize pixel values from [0, 255] to [0, 1] — helps the network train
# faster and more stably.
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Labels come in as integers (0-9); convert to one-hot vectors for
# categorical_crossentropy loss.
y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=10)

# Carve out a validation set from training data (used to monitor overfitting)
val_split = 0.1
val_size = int(len(x_train) * val_split)
x_val, y_val = x_train[:val_size], y_train_cat[:val_size]
x_train_final, y_train_final = x_train[val_size:], y_train_cat[val_size:]

print(f"\nFinal split -> train: {len(x_train_final)}, val: {len(x_val)}, test: {len(x_test)}")

# ---------------------------------------------------------------------------
# 3. BUILD THE CNN
# ---------------------------------------------------------------------------
def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ---------------------------------------------------------------------------
# 4. (OPTIONAL BUT RECOMMENDED) DATA AUGMENTATION
# ---------------------------------------------------------------------------
# Randomly flips/shifts/rotates training images each epoch, which acts like
# "free" extra data and reduces overfitting.
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train_final)

# ---------------------------------------------------------------------------
# 5. TRAIN
# ---------------------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_accuracy'),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, monitor='val_loss')
]

EPOCHS = 40
BATCH_SIZE = 64

print("\nTraining the model...")
history = model.fit(
    datagen.flow(x_train_final, y_train_final, batch_size=BATCH_SIZE),
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---------------------------------------------------------------------------
# 6. PLOT TRAINING CURVES
# ---------------------------------------------------------------------------
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_title('Accuracy over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_title('Loss over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=100)
    plt.close()
    print("Saved training_curves.png")

plot_history(history)

# ---------------------------------------------------------------------------
# 7. EVALUATE ON TEST SET
# ---------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")
print(f"Test loss:     {test_loss:.4f}")

# Predictions
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test.flatten()

# Classification report (precision, recall, F1 per class)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.close()
print("Saved confusion_matrix.png")

# ---------------------------------------------------------------------------
# 8. SAVE THE MODEL
# ---------------------------------------------------------------------------
model.save('cnn_cifar10_model.keras')
print("\nModel saved as cnn_cifar10_model.keras")

# ---------------------------------------------------------------------------
# 9. VISUALIZE SOME PREDICTIONS (correct vs incorrect)
# ---------------------------------------------------------------------------
def plot_predictions(x, y_true, y_pred, class_names, n=10):
    plt.figure(figsize=(12, 5))
    for i in range(n):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        plt.title(f"True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}",
                   color=color, fontsize=9)
        plt.axis('off')
    plt.suptitle("Predictions on Test Set (green=correct, red=wrong)")
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=100)
    plt.close()
    print("Saved sample_predictions.png")

plot_predictions(x_test, y_true, y_pred, class_names)

print("\nDone! Check the saved PNGs and the trained model file.")
