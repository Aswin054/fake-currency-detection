import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

# Generator Model
def build_generator():
    model = Sequential([
        Dense(128, input_dim=100, activation=LeakyReLU(alpha=0.2)),
        Dense(256, activation=LeakyReLU(alpha=0.2)),
        Dense(512, activation=LeakyReLU(alpha=0.2)),
        Dense(28*28, activation='tanh'),
        Reshape((28, 28))
    ])
    return model

# Generate Fake Images
generator = build_generator()
random_noise = np.random.normal(0, 1, (5, 100))  # Generate 5 random noise vectors
fake_images = generator.predict(random_noise)

# Display Generated Images
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(fake_images[i], cmap='gray')
    plt.axis('off')
plt.show()
