import cv2
import numpy as np

def pad_image(image, percent):
    # Calculate padding size
    pad_width = int(image.shape[1] * percent / 100)
    pad_height = int(image.shape[0] * percent / 100)
    
    # Pad the image
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_REPLICATE)
    
    return padded_image

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image

def crop_center(image, crop_width, crop_height):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate crop area
    start_x = max(width // 2 - crop_width // 2, 0)
    start_y = max(height // 2 - crop_height // 2, 0)
    end_x = min(start_x + crop_width, width)
    end_y = min(start_y + crop_height, height)

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def fill_padding(image, original_image):
    # Get image dimensions
    height, width = image.shape[:2]

    # Get the color of the nearby pixels for filling
    fill_color = np.mean(original_image[1:height-1, 1:width-1], axis=(0, 1))

    # Fill the padding with the color of nearby pixels
    image[:1, :] = fill_color
    image[-1:, :] = fill_color
    image[:, :1] = fill_color
    image[:, -1:] = fill_color

    return image

# Load the image
image = cv2.imread('input_image.jpg')

# Specify the rotation angle (in degrees, clockwise)
angle = 45

# Pad the image by 10%
padded_image = pad_image(image, 10)

# Rotate the padded image
rotated_image = rotate_image(padded_image, angle)

# Crop the rotated image to original size from the center
cropped_image = crop_center(rotated_image, image.shape[1], image.shape[0])

# Fill the padding with nearby pixel colors
filled_image = fill_padding(cropped_image, image)

# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated and Padded Image', filled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
