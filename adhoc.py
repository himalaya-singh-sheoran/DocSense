import cv2
import numpy as np
import random

def degrade_image(image):
    # Add noise
    noise = np.random.normal(loc=0, scale=25, size=image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)

    # Apply JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]  # Adjust the quality parameter as needed (0-100)
    _, compressed_image = cv2.imencode('.jpg', noisy_image, encode_param)
    degraded_image = cv2.imdecode(compressed_image, 1)

    return degraded_image

# Load the image
image = cv2.imread('input_image.jpg')

# Degrade the image
degraded_image = degrade_image(image)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Degraded Image', degraded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('degraded_image.jpg', degraded_image)
