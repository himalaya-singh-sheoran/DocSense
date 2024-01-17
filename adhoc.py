import cv2
import numpy as np

def remove_straight_lines(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a mask to fill detected lines
    mask = np.zeros_like(edges)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 2)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Bitwise AND the original image with the inverted mask
    result = cv2.bitwise_and(image, image, mask=mask_inv)

    # Display the result
    cv2.imshow('Image without straight lines', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the actual path to your image
remove_straight_lines('your_image_path.jpg')
