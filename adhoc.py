from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np

# Load an image
image = np.zeros((300, 500, 3), dtype=np.uint8)

# Convert the image to RGB (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a PIL Image from the RGB image array
pil_image = Image.fromarray(image_rgb)

# Choose a font (Arial, size 40)
font = ImageFont.truetype("Arial.ttf", 40)

# Create a PIL ImageDraw object
draw = ImageDraw.Draw(pil_image)

# Draw text on the image
draw.text((50, 100), "Hello, OpenCV!", font=font, fill=(255, 255, 255))

# Convert the PIL Image back to an OpenCV image array
image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Display the image with text
cv2.imshow("Image with Text", image_with_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
