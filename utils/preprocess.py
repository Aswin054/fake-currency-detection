from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size (resize images to 128x128)
image_size = (128, 128)

# Function to preprocess a single image
def preprocess_image(image_path):
    """
    This function takes an image path, loads the image, resizes it to the target size,
    normalizes it, and returns the processed image ready for prediction or training.
    """
    # Load and resize the image to the target size
    img = image.load_img(image_path, target_size=image_size)
    
    # Convert image to numpy array and normalize (rescale pixel values to [0, 1])
    img_array = image.img_to_array(img) / 255.0
    
    # Add batch dimension (the model expects a batch, not just one image)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to augment the images during training (optional)
def create_train_datagen():
    """
    This function creates an ImageDataGenerator for training data with augmentation.
    You can add more augmentations based on your needs.
    """
    return ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values
        rotation_range=40,  # Randomly rotate images
        width_shift_range=0.2,  # Randomly shift images horizontally
        height_shift_range=0.2,  # Randomly shift images vertically
        shear_range=0.2,  # Randomly apply shear transformations
        zoom_range=0.2,  # Randomly zoom in and out
        horizontal_flip=True,  # Randomly flip images horizontally
        fill_mode='nearest'  # Fill in missing pixels after transformations
    )

# Function to create an ImageDataGenerator for testing data (no augmentation)
def create_test_datagen():
    """
    This function creates an ImageDataGenerator for test data.
    The test data will be rescaled but won't undergo augmentation.
    """
    return ImageDataGenerator(rescale=1./255)
