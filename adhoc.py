import cv2
import numpy as np

def generate_text_mask(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of text regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for text regions
    text_mask = np.zeros_like(gray_image)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.drawContours(text_mask, [contour], 0, (255, 255, 255), -1)

    return text_mask

# Load the enhanced image
enhanced_image = cv2.imread('enhanced_image.jpg')

# Generate the mask for enhanced text regions
text_mask = generate_text_mask(enhanced_image)

# Display the mask
cv2.imshow('Text Mask', text_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
cv2.imwrite('text_mask.jpg', text_mask)
