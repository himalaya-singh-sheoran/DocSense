import cv2
import numpy as np

def merge_all_boxes(image_path, overlap_threshold=0.3):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store bounding boxes
    bounding_boxes = []

    # Get bounding boxes from contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))

    # Function to calculate IoU (Intersection over Union) between two boxes
    def calculate_iou(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # Calculate intersection area
        inter_area = max(0, min(x2, x4) - max(x1, x3) + 1) * max(0, min(y2, y4) - max(y1, y3) + 1)

        # Calculate union area
        area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
        area_box2 = (x4 - x3 + 1) * (y4 - y3 + 1)
        union_area = area_box1 + area_box2 - inter_area

        # Calculate IoU
        iou = inter_area / union_area

        return iou

    # Apply non-maximum suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bounding_boxes, [1.0] * len(bounding_boxes), 0, overlap_threshold)

    # Draw merged bounding boxes on the original image
    for i in indices:
        box = bounding_boxes[i[0]]
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the image with merged bounding boxes
    cv2.imshow('Image with Merged Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the actual path to your image
# Adjust the overlap_threshold as needed
merge_all_boxes('your_image_path.jpg', overlap_threshold=0.3)

