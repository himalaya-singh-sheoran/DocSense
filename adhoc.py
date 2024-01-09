import cv2
import numpy as np
import random

def minimal_augmentation(image):
    # Randomly select an augmentation
    augmentation = random.choice(['flip', 'rotate', 'brightness', 'contrast', 'color_shift', 'noise', 'sharpen'])

    # Store original image dimensions
    rows, cols, _ = image.shape

    if augmentation == 'flip':
        # Flip the image horizontally
        augmented_image = cv2.flip(image, 1)  # Change the flip code (0, 1, or -1) for different directions
    elif augmentation == 'rotate':
        # Rotate the image by a small angle (-10 to 10 degrees)
        angle = random.randint(-10, 10)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        augmented_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    elif augmentation == 'brightness':
        # Adjust brightness by adding a random value to each pixel (brighten or darken)
        brightness_value = random.randint(-20, 20)
        augmented_image = np.clip(image.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)
    elif augmentation == 'contrast':
        # Adjust contrast by applying a random factor to the image
        contrast_value = random.uniform(0.8, 1.2)
        augmented_image = np.clip(image.astype(np.float32) * contrast_value, 0, 255).astype(np.uint8)
    elif augmentation == 'color_shift':
        # Slightly shift colors by changing the hue and saturation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(-10, 10)
        sat_shift = random.randint(-10, 10)
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + hue_shift, 0, 179)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + sat_shift, 0, 255)
        augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif augmentation == 'noise':
        # Add subtle noise to the image
        noise = np.random.normal(0, 5, (rows, cols, 3)).astype(np.uint8)
        augmented_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif augmentation == 'sharpen':
        # Apply sharpening filter to the image
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        augmented_image = cv2.filter2D(image, -1, kernel)

    return augmented_image

# Example usage:
# Load the image
input_image = cv2.imread('your_image.jpg')

# Apply minimal augmentation
augmented_image = minimal_augmentation(input_image)

# Display the original and augmented images
cv2.imshow('Original Image', input_image)
cv2.imshow('Augmented Image', augmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

