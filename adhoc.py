import cv2
import numpy as np

def generate_text_mask(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Invert the binary image to ensure text regions are white
    inverted_binary = cv2.bitwise_not(binary)

    # Apply morphological closing to fill small gaps in text regions
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(inverted_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Invert the resulting image again to make the text regions black
    final_text_mask = cv2.bitwise_not(closed_image)

    return final_text_mask

# Load the image
image = cv2.imread('input_image.jpg')

# Generate the mask for text regions
text_mask = generate_text_mask(image)

# Display the mask
cv2.imshow('Text Mask', text_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
cv2.imwrite('text_mask.jpg', text_mask)
