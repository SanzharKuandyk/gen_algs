import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

TASK = "mnist"   # "gaussian" or "mnist"

EPOCHS = 100
BATCH_SIZE = 64
Z_DIM = 5
LR = 0.001

if TASK == "gaussian":
    X_train = np.random.randn(2000, 1)
    DATA_DIM = 1
else:
    (X, _), _ = tf.keras.datasets.mnist.load_data()
    X_train = X.reshape(-1, 784).astype("float32") / 255.0
    X_train = X_train[:5000]
    DATA_DIM = 784

def build_generator():
    return Sequential([
        Dense(16, input_dim=Z_DIM, activation="relu"),
        Dense(DATA_DIM, activation="sigmoid" if TASK == "mnist" else "linear")
    ])

def build_discriminator():
    model = Sequential([
        Dense(16, input_dim=DATA_DIM, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=Adam(LR))
    return model

def build_gan(G, D):
    D.trainable = False
    z = Input(shape=(Z_DIM,))
    out = D(G(z))
    model = Model(z, out)
    model.compile(loss="binary_crossentropy", optimizer=Adam(LR))
    return model

G = build_generator()
D = build_discriminator()
GAN = build_gan(G, D)

def visualize():
    if TASK == "gaussian":
        z = np.random.randn(1000, Z_DIM)
        g = G(z).numpy().flatten()
        t = np.random.randn(1000)
        plt.hist(g, bins=40, density=True, alpha=0.7, label="GAN", color="blue")
        plt.hist(t, bins=40, density=True, alpha=0.7, label="True", color="orange")
        plt.legend()
        plt.show()
    else:
        z = np.random.randn(5, Z_DIM)
        imgs = G(z).numpy().reshape(-1, 28, 28)
        plt.figure(figsize=(6,2))
        for i in range(5):
            plt.subplot(1,5,i+1)
            plt.imshow(imgs[i], cmap="gray")
            plt.axis("off")
        plt.show()

n_batches = X_train.shape[0] // BATCH_SIZE

for epoch in range(EPOCHS):
    for _ in range(n_batches):
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        real = X_train[idx]

        z = np.random.randn(BATCH_SIZE, Z_DIM)
        fake = G(z, training=False)

        X = np.vstack((real, fake))
        y = np.zeros(2 * BATCH_SIZE)
        y[:BATCH_SIZE] = 0.9

        D.train_on_batch(X, y)

        z = np.random.randn(BATCH_SIZE, Z_DIM)
        GAN.train_on_batch(z, np.ones(BATCH_SIZE))

    if epoch % 20 == 0:
        print("Epoch", epoch)
        visualize()
