import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, utils

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus[0].name)
else:
    print("No GPU detected, running on CPU")

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test  = utils.to_categorical(y_test, NUM_CLASSES)

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

optimizer = optimizers.Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training model
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=15,
    validation_split=0.1,
    shuffle=True
)

# evaluation
loss, acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {acc:.4f}")

# plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

CLASSES = np.array([
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
])

preds = model.predict(x_test)
pred_labels = CLASSES[np.argmax(preds, axis=1)]
true_labels = CLASSES[np.argmax(y_test, axis=1)]

n_to_show = 10
indices = np.random.choice(len(x_test), n_to_show, replace=False)

plt.figure(figsize=(15, 3))
for i, idx in enumerate(indices):
    plt.subplot(1, n_to_show, i + 1)
    plt.imshow(x_test[idx])
    plt.axis('off')
    plt.title(f"P: {pred_labels[idx]}\nT: {true_labels[idx]}", fontsize=9)
plt.show()
