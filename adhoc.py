import cv2
import numpy as np

def enhance_image(image):
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB channels
    l, a, b = cv2.split(lab_image)

    # Apply contrast stretching
    l_stretched = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Apply histogram equalization
    l_equalized = cv2.equalizeHist(l_stretched)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l_stretched)

    # Merge LAB channels
    enhanced_lab_image = cv2.merge((l_clahe, a, b))

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)

    return enhanced_image

# Load the image
image = cv2.imread('input_image.jpg')

# Enhance the image
enhanced_image = enhance_image(image)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('enhanced_image.jpg', enhanced_image)
