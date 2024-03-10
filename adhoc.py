import cv2
import numpy as np

# Read the input image
image = cv2.imread('input_image.jpg')

# Read the mask (binary image)
mask = cv2.imread('mask_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the mask to extract the region of interest
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Convert the masked image to a list of RGB values
pixels = masked_image.reshape((-1, 3))
pixels = pixels[mask.flatten() > 0]

# Calculate the histogram of colors in the masked region
histogram = cv2.calcHist([pixels], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])

# Find the index of the most frequent color in the histogram
most_frequent_color_index = np.unravel_index(np.argmax(histogram), histogram.shape)

# Convert the index to BGR color format
most_frequent_color_bgr = np.array(most_frequent_color_index[::-1], dtype=np.uint8)

print("Most frequent color (BGR):", most_frequent_color_bgr)
