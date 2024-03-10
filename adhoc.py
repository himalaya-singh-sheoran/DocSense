import cv2
import numpy as np
import random

# List of system font names
font_names = [
    "Arial",
    "Times New Roman",
    "Courier New",
    "Verdana",
    "Tahoma",
    "Calibri",
    "Georgia",
    "Comic Sans MS",
    "Impact",
    "Trebuchet MS"
]

# Load an image
image = np.zeros((300, 500, 3), dtype=np.uint8)

# Randomly choose a font name
selected_font = random.choice(font_names)

# Render text using the selected font
cv2.putText(image, 'Hello, OpenCV!', (50, 100), selected_font, 1, (255, 255, 255), 2)

# Display the image with text
cv2.imshow('Image with Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
