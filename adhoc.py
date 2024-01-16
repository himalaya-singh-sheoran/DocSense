import cv2
import numpy as np

def detect_text_swt(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Stroke Width Transform
    swt = cv2.ximgproc.createStrokesWTSigma()
    swt_map = swt.computeSWT(gray)

    # Threshold the result to get text regions
    _, swt_thresholded = cv2.threshold(swt_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(swt_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the actual path to your image
detect_text_swt('your_image_path.jpg')
