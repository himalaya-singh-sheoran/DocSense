import cv2
import numpy as np

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Find non-zero pixels in the rotated image
    non_zero_pixels = cv2.findNonZero(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY))

    # Get the bounding box of the non-zero pixels
    x, y, w, h = cv2.boundingRect(non_zero_pixels)

    # Crop the rotated image to the bounding box
    cropped_image = rotated_image[y:y+h, x:x+w]

    # Inpaint the black areas
    mask = np.zeros(cropped_image.shape[:2], np.uint8)
    mask[cropped_image == 0] = 255
    inpainted_image = cv2.inpaint(cropped_image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted_image

# Load the image
image = cv2.imread('input_image.jpg')

# Specify the rotation angle (in degrees, clockwise)
angle = 45

# Rotate the image
rotated_image = rotate_image(image, angle)

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated and Inpainted Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
