import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size and batch size
image_size = (128, 128)
batch_size = 32

# Data augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1./255)

# Load training dataset
train_data = train_datagen.flow_from_directory(
    'dataset/train/',  # Make sure the path is correct
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (real vs fake)
])

# ✅ Paste the compile() function **here**, before training starts
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10)

# Save the trained model
model.save('models/currency_model.keras')  # Saves in recommended Keras format

