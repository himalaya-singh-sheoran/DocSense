import cv2
import numpy as np

# Read the input image
image = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to binarize the image
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

# Create horizontal and vertical structuring elements
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

# Perform morphological operations to detect and remove horizontal lines
horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cleaned_image = cv2.subtract(binary, horizontal_lines)

# Perform morphological operations to detect and remove vertical lines
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cleaned_image = cv2.subtract(cleaned_image, vertical_lines)

# Invert the result
cleaned_image = 255 - cleaned_image

# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Result', cleaned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
