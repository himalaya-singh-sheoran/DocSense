import cv2
import numpy as np

def generate_corner_mask(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect corners using Harris corner detection
    corner_mask = np.zeros_like(gray, dtype=np.uint8)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Threshold to obtain strong corners
    corner_mask[dst > 0.01 * dst.max()] = 255

    return corner_mask

# Load the image
image = cv2.imread('input_image.jpg')

# Generate the mask using corners
corner_mask = generate_corner_mask(image)

# Display the mask
cv2.imshow('Corner Mask', corner_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
cv2.imwrite('corner_mask.jpg', corner_mask)
