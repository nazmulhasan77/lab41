import tensorflow as tf
print(tf.__version__)



FCNN

from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import Model
inputs = Input((2,)) #vector , not matrix
h1 = Dense(1,activation='relu')(inputs)
h2 = Dense(3,activation='relu')(h1)
h3 = Dense(2,activation='relu')(h2)
outputs = Dense(1, activation='softmax')(h3)
model = Model(inputs, outputs)
model.summary(show_trainable=True)


 Fully Connected Feedforward Neural Network (FCFNN) for solving the equation f(x) = 5x^2 +10x -2 

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# Polynomial function
def my_polynomial(x):
    return 5 * (x ** 2) + 10 * x - 2

# Generate dataset
def data_process():
    n = 10000
    x = np.random.randint(-100, 100, n)  # Random integers 0-99
    y = my_polynomial(x)
    x = x.reshape(-1, 1)  # Make it 2D for Keras
    y = y.reshape(-1, 1)
    return x, y

# Split dataset
def prepare_train_test_val():
    x, y = data_process()
    total_n = len(x)
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)

    trainX = x[:train_n]
    trainY = y[:train_n]

    valX = x[train_n:train_n+val_n]
    valY = y[train_n:train_n+val_n]

    testX = x[train_n+val_n:]
    testY = y[train_n+val_n:]

    return (trainX, trainY), (valX, valY), (testX, testY)

# Build the model
def build_model():
    inputs = Input((1,))
    h1 = Dense(8, activation='relu', name='Hidden_Layer_1')(inputs)
    h2 = Dense(16, activation='relu', name='Hidden_Layer_2')(h1)
    h3 = Dense(64, activation='relu', name='Hidden_Layer_3')(h2)
    h4 = Dense(128, activation='relu', name='Hidden_Layer_4')(h3)
    h5 = Dense(32, activation='relu', name='Hidden_Layer_5')(h4)
    h6 = Dense(8, activation='relu', name='Hidden_Layer_6')(h5)
    outputs = Dense(1, name='Output_Layer')(h6)
    
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

# Main function
def main():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_test_val()

    # Train the model
    history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=100, batch_size=32, verbose=1)

    # Evaluate on test data
    test_loss = model.evaluate(testX, testY)
    print(f"\nTest Mean Squared Error: {test_loss:.4f}")

    # Predict on test data
    y_pred = model.predict(testX)

    # Plot original vs predicted
    plt.figure(figsize=(8,5))
    plt.scatter(testX, testY, label="Original f(x)", alpha=0.6)
    plt.scatter(testX, y_pred, label="Predicted f(x)", alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Original vs Predicted Function")
    plt.legend()
    plt.savefig("fx_prediction.png", dpi=300)  # save for LaTeX
    plt.show()

    # Plot training history
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.legend()
    plt.savefig("training_history.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()

CNN MNIST

import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. Load and preprocess data
# -------------------------------
def data_process():
    (trainX, trainY), (testX, testY) = mnist.load_data()

    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0

    # Add channel dimension
    trainX = trainX[..., tf.newaxis]
    testX = testX[..., tf.newaxis]

    return (trainX, trainY), (testX, testY)


# -------------------------------
# 2. Prepare train/val/test split
# -------------------------------
def prepare_train_test_val():
    (trainX, trainY), (testX, testY) = data_process()

    total_n = len(trainX)
    val_n = int(total_n * 0.1)

    valX = trainX[:val_n]
    valY = trainY[:val_n]

    trainX = trainX[val_n:]
    trainY = trainY[val_n:]

    return (trainX, trainY), (valX, valY), (testX, testY)


# -------------------------------
# 3. Build CNN model
# -------------------------------
def build_model():
    inputs = Input((28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model


# -------------------------------
# 4. Main function
# -------------------------------
def main():
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_test_val()

    history = model.fit(trainX, trainY,
                        validation_data=(valX, valY),
                        epochs=5,
                        batch_size=32,
                        verbose=1)

    # Evaluate
    test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # -------------------------------
    # Plot training history
    # -------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.savefig("training_accuracy.png", dpi=300)
    plt.show()

    # -------------------------------
    # Predictions
    # -------------------------------
    predictions = model.predict(testX)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(testX[i], cmap='gray')
        plt.title(f"Pred: {predicted_labels[i]}\nTrue: {testY[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=300)
    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()


CNN ODD EVEN

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time, random

class_labels = ['Even', 'Odd']

# -------------------------------
# 1. Load & preprocess data
# -------------------------------
def data_process():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Convert digits → Even(0) / Odd(1)
    y_train = np.array([0 if d % 2 == 0 else 1 for d in y_train])
    y_test  = np.array([0 if d % 2 == 0 else 1 for d in y_test])

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    # One-hot encoding
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat  = to_categorical(y_test, 2)

    return (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat)


# -------------------------------
# 2. Prepare train/val/test split
# -------------------------------
def prepare_train_val_test():
    (x_train, y_train, y_train_cat), (x_test, y_test, y_test_cat) = data_process()

    x_train, x_val, y_train_cat, y_val_cat = train_test_split(
        x_train, y_train_cat, test_size=0.2, random_state=42
    )

    return (x_train, y_train_cat), (x_val, y_val_cat), (x_test, y_test, y_test_cat)


# -------------------------------
# 3. Build CNN model
# -------------------------------
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# -------------------------------
# 4. Main function
# -------------------------------
def main():
    model = build_model()
    (trainX, trainY), (valX, valY), (testX, testY, testY_cat) = prepare_train_val_test()

    # -------------------------------
    # Training
    # -------------------------------
    start_time = time.time()

    history = model.fit(
        trainX, trainY,
        epochs=10,
        batch_size=64,
        validation_data=(valX, valY)
    )

    train_time = time.time() - start_time
    print(f"\n Training completed in {train_time:.2f} sec")

    # -------------------------------
    # Evaluation
    # -------------------------------
    start_test = time.time()
    test_loss, test_acc = model.evaluate(testX, testY_cat)
    test_time = time.time() - start_test

    print(f"\n Testing time: {test_time:.2f} sec")
    print(f" Test Accuracy: {test_acc*100:.2f}%")

    # -------------------------------
    # Plot Accuracy
    # -------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig("odd_even_accuracy.png", dpi=300)
    plt.show()

    # -------------------------------
    # Plot Loss
    # -------------------------------
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig("odd_even_loss.png", dpi=300)
    plt.show()

    # -------------------------------
    # Predict 10 Random Images
    # -------------------------------
    indices = random.sample(range(len(testX)), 10)

    plt.figure(figsize=(15,8))
    correct = 0

    for i, idx in enumerate(indices):
        img = testX[idx]
        true_label = class_labels[testY[idx]]

        pred = model.predict(np.expand_dims(img,0), verbose=0)
        pred_class = class_labels[np.argmax(pred)]
        conf = np.max(pred)

        if pred_class == true_label:
            correct += 1

        plt.subplot(2,5,i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"P:{pred_class}\nT:{true_label}\n{conf:.2f}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"\n Correct: {correct}/10 ({correct*10}%)")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    main()

Assignment 2
Fully Connected Feed-Forward Neural Network (FCFNN)

Architecture: Input(8) → 4 → 8 → 4 → 10(Output)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def build_model():

    inputs = Input(shape=(8,), name="input_layer")

    h1 = Dense(4, activation='relu', name='hidden_layer1')(inputs)
    h2 = Dense(8, activation='relu', name='hidden_layer2')(h1)
    h3 = Dense(4, activation='relu', name='hidden_layer3')(h2)
    outputs = Dense(10, activation='softmax', name='output_layer')(h3)
    model = Model(inputs, outputs, name="FCFNN_Functional")


    return model

def main():

    model = build_model()
    model.summary(show_trainable=True)

if __name__ == "__main__":
    main()


Assignment 3
Building FCFNNs for solving the following equations:

i. y = 5x + 10

ii. y = 3x2 + 5x + 10

iii. y = 4x3 + 3x2 + 5x + 10

• preparing a training set, a validation set and a test set for the above

equations.

• training and testing FCFNNs using your prepared data.

• plotting original y and ‘predicted y’.

• explaining the effect of ”power of an independent variable

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Generate Data
# --------------------------------------------------

def generate_data(power):

    x = np.linspace(-10, 10, 2000)

    if power == 1:
        y = 5*x + 10

    elif power == 2:
        y = 3*(x**2) + 5*x + 10

    elif power == 3:
        y = 4*(x**3) + 3*(x**2) + 5*x + 10

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return train_test_split(x, y, test_size=0.3, random_state=42)


# --------------------------------------------------
# Build Model
# --------------------------------------------------

def build_model():

    inputs = Input(shape=(1,), name="input_layer")

    h1 = Dense(32, activation='relu')(inputs)
    h2 = Dense(32, activation='relu')(h1)
    h3 = Dense(16, activation='relu')(h2)

    outputs = Dense(1, activation='linear')(h3)

    model = Model(inputs, outputs, name="FCFNN")

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=0
    )


# --------------------------------------------------
# Plot Results
# --------------------------------------------------

def plot_results(model, x, y, title, filename):

    y_pred = model.predict(x)

    plt.figure(figsize=(8,6))
    plt.scatter(x, y, label="Original y")
    plt.scatter(x, y_pred, label="Predicted y")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.savefig(filename)
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for power in [1, 2, 3]:

        print("\n===================================")
        print(f"Training for Power {power}")
        print("===================================")

        x_train, x_test, y_train, y_test = generate_data(power)

        model = build_model()

        model.summary()

        train_model(model, x_train, y_train)

        plot_results(
            model,
            x_test,
            y_test,
            title=f"Polynomial Power {power}",
            filename=f"power_{power}.png"
        )


if __name__ == "__main__":
    main()


Assignment 4
Building an FCFNN based classifier according to your preferences about

the number of hidden layers and neurons in the hidden layers.

• training and testing your FCFNN based classifier using the:

i. Fashion MNIST dataset.

ii. MNIST English dataset.

iii. CIFAR-10 dataset.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(dataset_name):

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build FCFNN Model
# --------------------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape, name="input_layer")

    x = Flatten()(inputs)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name="FCFNN_Classifier")

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Loss
# --------------------------------------------------

def plot_loss(history, name):

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss Curve")
    plt.legend()

    plt.savefig(f"{name}_loss.png")
    plt.show()


# --------------------------------------------------
# Plot 10 Predictions
# --------------------------------------------------

def plot_predictions(model, x_test, y_test, name):

    predictions = model.predict(x_test[:10])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10,4))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i], cmap='gray' if len(x_test.shape)==3 else None)
        plt.title(f"P:{predicted_labels[i]}\nT:{y_test[i]}")
        plt.axis('off')

    plt.suptitle(f"{name} Predictions")
    plt.savefig(f"{name}_predictions.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for dataset in ["mnist", "fashion", "cifar10"]:

        print("\n====================================")
        print(f"Training on {dataset.upper()}")
        print("====================================")

        x_train, x_test, y_train, y_test = load_data(dataset)

        model = build_model(input_shape=x_train.shape[1:])

        model.summary()

        history = train_model(model, x_train, y_train)

        model.evaluate(x_test, y_test, verbose=1)

        plot_loss(history, dataset)

        plot_predictions(model, x_test, y_test, dataset)


if __name__ == "__main__":
    main()
Assignment 5
Building a Convolutional Neural Network (CNN) based 10 class classifier

• training and testing the classifier using the:

i. Fashion MNIST dataset.

ii. MNIST English dataset.

iii. CIFAR-10 dataset.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data(dataset_name):

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    elif dataset_name == "fashion":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build CNN Model
# --------------------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape, name="input_layer")

    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name="CNN_Classifier")

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Loss
# --------------------------------------------------

def plot_loss(history, name):

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss Curve")
    plt.legend()

    plt.savefig(f"{name}_cnn_loss.png")
    plt.show()


# --------------------------------------------------
# Plot 10 Predictions
# --------------------------------------------------

def plot_predictions(model, x_test, y_test, name):

    predictions = model.predict(x_test[:10])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10,4))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap='gray')
        plt.title(f"P:{predicted_labels[i]}\nT:{y_test[i]}")
        plt.axis('off')

    plt.suptitle(f"{name} CNN Predictions")
    plt.savefig(f"{name}_cnn_predictions.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    for dataset in ["mnist", "fashion"]:

        print("\n====================================")
        print(f"Training CNN on {dataset.upper()}")
        print("====================================")

        x_train, x_test, y_train, y_test = load_data(dataset)

        model = build_model(input_shape=x_train.shape[1:])

        model.summary()

        history = train_model(model, x_train, y_train)

        model.evaluate(x_test, y_test, verbose=1)

        plot_loss(history, dataset)

        plot_predictions(model, x_test, y_test, dataset)


if __name__ == "__main__":
    main()


Assignment 8
Build a CNN based classifier having architecture similar to the classical VGG16.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build VGG16-like Model
# --------------------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape, name="input_layer")

    # Block 1
    x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 2
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 3
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)

    # Fully Connected Layers (VGG style)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name="VGG16_Like")

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Loss
# --------------------------------------------------

def plot_loss(history):

    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("VGG16-like Loss Curve")
    plt.legend()

    plt.savefig("vgg16_like_loss.png")
    plt.show()


# --------------------------------------------------
# Plot 10 Predictions
# --------------------------------------------------

def plot_predictions(model, x_test, y_test):

    predictions = model.predict(x_test[:10])
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(10,4))

    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap='gray')
        plt.title(f"P:{predicted_labels[i]}\nT:{y_test[i]}")
        plt.axis('off')

    plt.suptitle("VGG16-like Predictions")
    plt.savefig("vgg16_like_predictions.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    model = build_model(input_shape=x_train.shape[1:])

    model.summary()

    history = train_model(model, x_train, y_train)

    model.evaluate(x_test, y_test, verbose=1)

    plot_loss(history)

    plot_predictions(model, x_test, y_test)


if __name__ == "__main__":
    main()
Assignment 10
Write a report in pdf format using any Latex system after:

● training a binary classifier, based on the pre-trained VGG16, by transfer learning

and fine tuning.

● showing the effect of fine-tuning:

i. whole pre-trained VGG16

ii. partial pre-trained VGG16


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# --------------------------------------------------
# Load Dataset (Binary: Cats vs Dogs)
# --------------------------------------------------

def load_data():

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keep only class 3 (cat) and 5 (dog)
    train_filter = np.where((y_train==3) | (y_train==5))[0]
    test_filter = np.where((y_test==3) | (y_test==5))[0]

    x_train = x_train[train_filter]
    y_train = y_train[train_filter]
    x_test = x_test[test_filter]
    y_test = y_test[test_filter]

    y_train = (y_train==5).astype(int)
    y_test = (y_test==5).astype(int)

    x_train = tf.image.resize(x_train, (224,224))
    x_test = tf.image.resize(x_test, (224,224))

    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------
# Build Model
# --------------------------------------------------

def build_model(trainable_layers=None):

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

    if trainable_layers is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False

    inputs = Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    return model


# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(model, x_train, y_train):

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    return history


# --------------------------------------------------
# Plot Results
# --------------------------------------------------

def plot_history(history, name):

    # Loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{name} Loss")
    plt.legend()
    plt.savefig(f"{name}_loss.png")
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f"{name} Accuracy")
    plt.legend()
    plt.savefig(f"{name}_accuracy.png")
    plt.show()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    x_train, x_test, y_train, y_test = load_data()

    # -------------------------------
    # Case 1: Freeze Whole VGG16
    # -------------------------------
    print("\n=== Transfer Learning (Freeze All Layers) ===")
    model1 = build_model(trainable_layers=None)
    history1 = train_model(model1, x_train, y_train)
    model1.evaluate(x_test, y_test)
    plot_history(history1, "freeze_all")


    # -------------------------------
    # Case 2: Partial Fine-Tuning
    # -------------------------------
    print("\n=== Partial Fine-Tuning (Last 4 Layers Trainable) ===")
    model2 = build_model(trainable_layers=4)
    history2 = train_model(model2, x_train, y_train)
    model2.evaluate(x_test, y_test)
    plot_history(history2, "partial_finetune")


    # -------------------------------
    # Case 3: Fine-Tune Whole VGG16
    # -------------------------------
    print("\n=== Fine-Tune Whole VGG16 ===")
    model3 = build_model(trainable_layers=len(VGG16().layers))
    history3 = train_model(model3, x_train, y_train)
    model3.evaluate(x_test, y_test)
    plot_history(history3, "full_finetune")


if __name__ == "__main__":
    main()


import tensorflow as tf
print(tf.__version__)


------------

FCNN

from tensorflow.keras.layers import Input,Dense
from tensorflow.keras import Model
inputs = Input((2,)) #vector , not matrix
h1 = Dense(1,activation='relu')(inputs)
h2 = Dense(3,activation='relu')(h1)
h3 = Dense(2,activation='relu')(h2)
outputs = Dense(1, activation='softmax')(h3)
model = Model(inputs, outputs)
model.summary(show_trainable=True)

-----------

CNN MNIST

#Importing Libraries
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten,Conv2D, MaxPooling2D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# 2. Normalize and reshape data
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# CNN expects (height, width, channels)
trainX = trainX[..., tf.newaxis]  
testX = testX[..., tf.newaxis]

#CNN Model Create
inputs = Input((28, 28, 1))  # include channel dimension for grayscale
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
# Train Model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, validation_split=0.1, epochs=5, batch_size=32, verbose=1)

# Evaluate on test set
test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
print(f"\n Test Accuracy: {test_acc:.4f}")


# Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.savefig("training_accuracy.png")
plt.legend()
plt.show()

# Predict on test set
predictions = model.predict(testX)
predicted_labels = np.argmax(predictions, axis=1)

# Show 20 test images with predicted labels in a grid
plt.figure(figsize=(12, 8))

for i in range(10):
    plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
    plt.imshow(testX[i], cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {testY[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("test_predictions.png")
plt.show()

-----------

#Importing Libraries
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Flatten,Conv2D, MaxPooling2D, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
# Load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
# 2. Normalize and reshape data
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# CNN expects (height, width, channels)
trainX = trainX[..., tf.newaxis]  
testX = testX[..., tf.newaxis]

#CNN Model Create
inputs = Input((28, 28, 1))  # include channel dimension for grayscale
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs, outputs)
# Train Model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(trainX, trainY, validation_split=0.1, epochs=5, batch_size=32, verbose=1)

# Evaluate on test set
test_loss, test_acc = model.evaluate(testX, testY, verbose=0)
print(f"\n Test Accuracy: {test_acc:.4f}")


# Plot training history
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.xlabel("No of Epochs")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.savefig("training_accuracy.png")
plt.legend()
plt.show()

# Predict on test set
predictions = model.predict(testX)
predicted_labels = np.argmax(predictions, axis=1)

# Show 20 test images with predicted labels in a grid
plt.figure(figsize=(12, 8))

for i in range(10):
    plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
    plt.imshow(testX[i], cmap='gray')
    plt.title(f"Pred: {predicted_labels[i]}\nTrue: {testY[i]}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("test_predictions.png")
plt.show()

-------------

# ========================================
# CIFAR-10 CNN Classifier with Data Augmentation
# and Visualization of Predictions
# ========================================

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random

# 1️ Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # normalize

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Original training samples: {x_train.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# 2️ Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(x_train)

# Data augmentation doesn’t *increase* the dataset size on disk,
# but generates new images in memory during each epoch.
# To demonstrate, let’s “generate” one epoch of augmented data:
augmented_images, _ = next(datagen.flow(x_train, y_train, batch_size=len(x_train), shuffle=False))
print(f"Augmented images generated in memory: {augmented_images.shape[0]}")

# 3️ Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4️ Train Model with Data Augmentation
batch_size = 64
epochs = 20

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train)//batch_size,
    epochs=epochs,
    verbose=1
)

# 5️ Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")

# 6️ Plot Accuracy & Loss Curves
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 7️ Predict and Visualize Results
num_images = 12
indices = random.sample(range(len(x_test)), num_images)
images = x_test[indices]
true_labels = y_test[indices].flatten()

predictions = model.predict(images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12,8))
for i in range(num_images):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i])
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    plt.title(f"P: {class_names[predicted_labels[i]]}\nT: {class_names[true_labels[i]]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()
-------------

 Fully Connected Feedforward Neural Network (FCFNN) for solving the equation f(x) = 5x^2 +10x -2 

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import numpy as np
import matplotlib.pyplot as plt

# Polynomial function
def my_polynomial(x):
    return 5 * (x ** 2) + 10 * x - 2

# Generate dataset
def data_process():
    n = 10000
    x = np.random.randint(-100, 100, n)  # Random integers 0-99
    y = my_polynomial(x)
    x = x.reshape(-1, 1)  # Make it 2D for Keras
    y = y.reshape(-1, 1)
    return x, y

# Split dataset
def prepare_train_test_val():
    x, y = data_process()
    total_n = len(x)
    train_n = int(total_n * 0.7)
    val_n = int(total_n * 0.1)

    trainX = x[:train_n]
    trainY = y[:train_n]

    valX = x[train_n:train_n+val_n]
    valY = y[train_n:train_n+val_n]

    testX = x[train_n+val_n:]
    testY = y[train_n+val_n:]

    return (trainX, trainY), (valX, valY), (testX, testY)

# Build the model
def build_model():
    inputs = Input((1,))
    h1 = Dense(8, activation='relu', name='Hidden_Layer_1')(inputs)
    h2 = Dense(16, activation='relu', name='Hidden_Layer_2')(h1)
    h3 = Dense(64, activation='relu', name='Hidden_Layer_3')(h2)
    h4 = Dense(128, activation='relu', name='Hidden_Layer_4')(h3)
    h5 = Dense(32, activation='relu', name='Hidden_Layer_5')(h4)
    h6 = Dense(8, activation='relu', name='Hidden_Layer_6')(h5)
    outputs = Dense(1, name='Output_Layer')(h6)
    
    model = Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

# Main function
def main():
    model = build_model()
    model.compile(optimizer='adam', loss='mse')

    (trainX, trainY), (valX, valY), (testX, testY) = prepare_train_test_val()

    # Train the model
    history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=100, batch_size=32, verbose=1)

    # Evaluate on test data
    test_loss = model.evaluate(testX, testY)
    print(f"\nTest Mean Squared Error: {test_loss:.4f}")

    # Predict on test data
    y_pred = model.predict(testX)

    # Plot original vs predicted
    plt.figure(figsize=(8,5))
    plt.scatter(testX, testY, label="Original f(x)", alpha=0.6)
    plt.scatter(testX, y_pred, label="Predicted f(x)", alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Original vs Predicted Function")
    plt.legend()
    plt.savefig("fx_prediction.png", dpi=300)  # save for LaTeX
    plt.show()

    # Plot training history
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training History")
    plt.legend()
    plt.savefig("training_history.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()


------------

# ==========================================
# Odd vs Even CNN Classifier (from MNIST)
# ==========================================

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import time, random

# ==============================
# 1️ Load MNIST Dataset
# ==============================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#C:\Users\<your_username>\.keras\datasets\mnist.npz
#~/.keras/datasets/mnist.npz

print("Original MNIST:")
print("Train:", x_train.shape, y_train.shape)
print("Test :", x_test.shape, y_test.shape)


# ==============================
# 2️ Convert digits → Odd/Even
# ==============================
# Even -> 0, Odd -> 1
y_train = np.array([0 if d % 2 == 0 else 1 for d in y_train])
y_test  = np.array([0 if d % 2 == 0 else 1 for d in y_test])

class_labels = ['Even', 'Odd']


# ==============================
# 3️ Preprocess Data
# =============================
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# One-hot encoding
y_train_cat = to_categorical(y_train, 2)
y_test_cat  = to_categorical(y_test, 2)

# Train/validation split
x_train, x_val, y_train_cat, y_val_cat = train_test_split(
    x_train, y_train_cat, test_size=0.2, random_state=42
)


# ==============================
# 4️ CNN Model Definition
# ==============================
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =============================
# 5️ Training
# ==============================
start_time = time.time()

history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=64,
    validation_data=(x_val, y_val_cat)
)

train_time = time.time() - start_time
print(f"\n✅ Training completed in {train_time:.2f} sec")


# ==============================
# 6️ Evaluate
# ==============================
start_test = time.time()

test_loss, test_acc = model.evaluate(x_test, y_test_cat)

test_time = time.time() - start_test

print(f"\n Testing time: {test_time:.2f} sec")
print(f" Test Accuracy: {test_acc*100:.2f}%")


# ==============================
# 7️ Plot Accuracy & Loss
# ==============================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.savefig("odd_even_accuracy.png")
plt.show()


plt.figure(figsize=(10,5))
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.savefig("odd_even_loss.png")
plt.show()


# ==============================
# 8️ Predict 10 Random Images
# ==============================
indices = random.sample(range(len(x_test)), 10)

plt.figure(figsize=(15,8))
correct = 0

for i, idx in enumerate(indices):

    img = x_test[idx]
    true_label = class_labels[y_test[idx]]

    pred = model.predict(np.expand_dims(img,0), verbose=0)
    pred_class = class_labels[np.argmax(pred)]
    conf = np.max(pred)

    if pred_class == true_label:
        correct += 1

    plt.subplot(2,5,i+1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"P:{pred_class}\nT:{true_label}\n{conf:.2f}")
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"\n Correct: {correct}/10 ({correct*10}%)")


# ==============================
# 9️ Save Model
# =============================
model.save("mnist_odd_even_cnn.h5")
print("\n Model saved as 'mnist_odd_even_cnn.h5'")


------------


Transfer Learning



from tensorflow.keras.applications import vgg16
model = vgg16.VGG16()
model.summary(show_trainable = True )

base_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3)) #input_shape=(224, 224, 3)
base_model.summary(show_trainable=True)

inputs = base_model.input
x=base_model.output
x = layers.Flatten()(x)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.Dense(2, activation='softmax')(x)

binary_model = models.Model(inputs=inputs, outputs=outputs)
binary_model.summary(show_trainable=True)

for layer in binary_model.layers[:-2]:
    layer.trainable = False
binary_model.summary(show_trainable=True)






