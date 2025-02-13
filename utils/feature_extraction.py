from skimage.feature import hog
from skimage import exposure
import numpy as np
import cv2

# Function to extract HOG features from an image
def extract_hog_features(image_path):
    """
    Extracts Histogram of Oriented Gradients (HOG) features from an image.
    """
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize image to a consistent size
    image_resized = cv2.resize(image, (128, 128))
    
    # Extract HOG features from the image
    features, hog_image = hog(image_resized, pixels_per_cell=(16, 16),
    cells_per_block=(2, 2), visualize=True, multichannel=False)
    
    # Enhance the HOG image (optional)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Return the features (flattened HOG features)
    return features


