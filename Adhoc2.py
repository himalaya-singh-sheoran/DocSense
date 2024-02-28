import cv2
import numpy as np
import random

def replace_text_regions(image, text_regions):
    # Load the OCR engine
    ocr_engine = cv2.text.OCRTesseract_create()

    # Loop through detected text regions
    for region in text_regions:
        x, y, w, h = region
        # Extract the text from the original image region
        roi = image[y:y+h, x:x+w]
        
        # Apply pre-processing if needed (e.g., thresholding)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Perform OCR on the region
        _, text = ocr_engine.detectAndDecode(roi)
        text = text.strip() if text is not None else ""
        
        # Get the color of the text region
        text_color = image[y, x].tolist()
        
        # Generate new text with the same number of characters
        new_text = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=len(text)))
        
        # Estimate font size to fit within the bounding box
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(new_text, font, font_scale, thickness)
        while text_width > w or text_height > h:
            font_scale -= 0.1
            (text_width, text_height), _ = cv2.getTextSize(new_text, font, font_scale, thickness)
        
        # Overlay new text onto the image
        color = text_color
        text_x = x + (w - text_width) // 2
        text_y = y + (h + text_height) // 2
        cv2.putText(image, new_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image

# Load the image
image = cv2.imread('input_image.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize MSER detector
mser = cv2.MSER_create()

# Detect text regions using MSER
text_regions, _ = mser.detectRegions(gray)

# Convert text regions to bounding boxes
text_regions = [cv2.boundingRect(region) for region in text_regions]

# Replace detected text regions with dynamically generated new text
image_with_new_text = replace_text_regions(image.copy(), text_regions)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Image with Dynamically Generated New Text', image_with_new_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('image_with_dynamically_generated_new_text.jpg', image_with_new_text)
