import cv2
import numpy as np

def generate_text_mask(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of text regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for text regions
    text_mask = np.zeros_like(gray)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(text_mask, (x, y), (x + w, y + h), (255), -1)

    return text_mask

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
