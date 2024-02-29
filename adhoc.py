import cv2
import numpy as np

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Determine the new dimensions of the rotated image
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_width = int(height * sin_theta + width * cos_theta)
    new_height = int(height * cos_theta + width * sin_theta)

    # Adjust the rotation matrix for the padding
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Pad the image to accommodate the rotated image
    padded_image = cv2.copyMakeBorder(image, int((new_height - height) / 2), int((new_height - height) / 2),
                                       int((new_width - width) / 2), int((new_width - width) / 2),
                                       borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Perform the rotation with bilinear interpolation
    rotated_image = cv2.warpAffine(padded_image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

    return rotated_image

# Load the image
image = cv2.imread('input_image.jpg')

# Specify the rotation angle (clockwise)
angle = 45

# Rotate the image
rotated_image = rotate_image(image, angle)

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
