import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import random

# Paths
real_currency_path = "dataset/train/real/"   # Folder containing real currency
fake_currency_path = "dataset/train/fake/"   # Folder to save generated fake images

# Create fake currency folder if it doesn't exist
os.makedirs(fake_currency_path, exist_ok=True)

# Load real currency images
real_images = os.listdir(real_currency_path)

def apply_random_transformation(image):
    """Applies random distortions to create fake-like images."""

    # Convert image to NumPy array (RGB format)
    img = np.array(image)

    # Ensure image is in uint8 format
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Apply random noise
    if random.random() > 0.5:
        noise = np.random.randint(0, 50, img.shape, dtype='uint8')
        img = cv2.add(img, noise)

    # Apply Gaussian blur
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Convert NumPy array back to PIL Image
    img_pil = Image.fromarray(img)

    # Apply brightness and contrast variations
    enhancer = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer.enhance(random.uniform(0.6, 1.4))

    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(random.uniform(0.6, 1.4))

    return img_pil  # Return PIL Image

# Generate fake samples
for i, img_name in enumerate(real_images):
    img_path = os.path.join(real_currency_path, img_name)
    
    try:
        img = Image.open(img_path).convert("RGB")  # Convert to RGB

        # Apply transformations
        transformed_img = apply_random_transformation(img)

        # Save the fake image
        fake_img_path = os.path.join(fake_currency_path, f"fake_{i}.jpg")
        transformed_img.save(fake_img_path)

        print(f"✅ Generated Fake Image: {fake_img_path}")

    except Exception as e:
        print(f"❌ Error processing {img_name}: {e}")

print(f"\n✅ Successfully generated {len(real_images)} fake currency samples!")

    


