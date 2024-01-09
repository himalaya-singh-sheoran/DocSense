import cv2
import numpy as np
import random

def minimal_augmentation(image):
    # Define a list of possible augmentations
    augmentations = ['flip', 'rotate', 'brightness', 'contrast', 'color_shift', 'noise', 'blur', 'sharpen', 'resize', 'crop']

    # Randomly select an augmentation
    selected_augmentation = random.choice(augmentations)

    # Store original image dimensions
    rows, cols, _ = image.shape

    # Make a copy of the original image for augmentation
    augmented_image = np.copy(image)

    if selected_augmentation == 'flip':
        # Flip the image horizontally
        augmented_image = cv2.flip(image, 1)  # Change the flip code (0, 1, or -1) for different directions
    elif selected_augmentation == 'rotate':
        # Rotate the image by a small angle (-5 to 5 degrees)
        angle = random.randint(-5, 5)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        augmented_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    elif selected_augmentation == 'brightness':
        # Adjust brightness by adding a small random value to each pixel
        brightness_value = random.randint(-10, 10)
        augmented_image = np.clip(image.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)
    elif selected_augmentation == 'contrast':
        # Adjust contrast by applying a slight random factor to the image
        contrast_value = random.uniform(0.95, 1.05)
        augmented_image = np.clip(image.astype(np.float32) * contrast_value, 0, 255).astype(np.uint8)
    elif selected_augmentation == 'color_shift':
        # Slightly shift colors by changing the hue and saturation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(-5, 5)
        sat_shift = random.randint(-5, 5)
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + hue_shift, 0, 179)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + sat_shift, 0, 255)
        augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif selected_augmentation == 'noise':
        # Add subtle noise to the image
        noise = np.random.normal(0, 3, (rows, cols, 3)).astype(np.uint8)
        augmented_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif selected_augmentation == 'blur':
        # Apply Gaussian blur with a small kernel size (3x3 or 5x5)
        kernel_size = random.choice([3, 5])
        augmented_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif selected_augmentation == 'sharpen':
        # Apply a small random sharpening filter
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        augmented_image = cv2.filter2D(image, -1, kernel_sharpening)
    elif selected_augmentation == 'resize':
        # Resize the image by a small percentage
        scale_factor = random.uniform(0.95, 1.05)
        resized_width = int(cols * scale_factor)
        resized_height = int(rows * scale_factor)
        augmented_image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    elif selected_augmentation == 'crop':
        # Randomly crop a region from the image (maintain at least 90% of the original image)
        crop_size = random.randint(int(min(rows, cols) * 0.9), min(rows, cols))
        start_row = random.randint(0, rows - crop_size)
        start_col = random.randint(0, cols - crop_size)
        augmented_image = image[start_row:start_row + crop_size, start_col:start_col + crop_size]

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

