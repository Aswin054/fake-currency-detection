import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Set image size (resize images to 128x128)
image_size = (128, 128)

# Load the trained model from the 'models/' folder
model = load_model('models/currency_model.keras')

# Create an ImageDataGenerator for test data (normalizing the pixel values)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the test data from the test folders (test/real and test/fake)
test_data = test_datagen.flow_from_directory(
    'dataset/test/',  # Path to your test data
    target_size=image_size,  # Resize images to 128x128
    batch_size=32,  # Number of images per batch
    class_mode='binary',  # Binary classification (real vs fake)
    shuffle=False  # Don't shuffle for evaluation
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data, steps=len(test_data))

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Optionally, make predictions on a single image (for testing purposes)
def predict_image(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict using the model
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print("Prediction: Fake Currency")
    else:
        print("Prediction: Real Currency")

# Example usage: Test with a single image
predict_image(r'C:\Users\Lenova\Desktop\fake currency\dataset\test\real\1.jpg')
  # Replace with the path to an image
