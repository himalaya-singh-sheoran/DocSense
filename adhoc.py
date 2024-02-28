import cv2
import numpy as np

def enhance_image(image):
    # Apply gamma correction
    gamma = 1.5
    corrected_image = np.uint8(cv2.pow(image / 255.0, gamma) * 255)

    # Apply histogram equalization
    gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)

    # Apply sharpening
    kernel_sharpening = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    sharpened_image = cv2.filter2D(equalized_image, -1, kernel_sharpening)

    return sharpened_image

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
