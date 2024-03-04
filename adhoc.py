import cv2
import numpy as np

# Read the input image
image = cv2.imread('input_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 50, 150)

# Perform morphological closing to fill in small gaps
kernel = np.ones((5, 5), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Invert the closed edges to create a mask
mask = 255 - closed_edges

# Apply the mask to the original image to remove the table boundaries
result = cv2.bitwise_and(image, image, mask=mask)

# Display the original and processed images
cv2.imshow('Original Image', image)
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
