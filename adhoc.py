import cv2
import numpy as np

def join_close_and_intersecting_boxes(image_path, proximity_threshold=20):
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

    # Function to check if two bounding boxes intersect
    def intersect(box1, box2):
        return not (box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1])

    # Function to check if two bounding boxes are close
    def close(box1, box2):
        center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
        center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
        distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
        return distance < proximity_threshold

    # Merge intersecting and close bounding boxes
    merged_boxes = []
    while bounding_boxes:
        current_box = bounding_boxes.pop(0)
        merged_box = list(current_box)
        merged = False
        for box in bounding_boxes:
            if intersect(merged_box, box) or close(merged_box, box):
                # Merge intersecting or close boxes
                merged_box[0] = min(merged_box[0], box[0])
                merged_box[1] = min(merged_box[1], box[1])
                merged_box[2] = max(merged_box[2], box[2])
                merged_box[3] = max(merged_box[3], box[3])
                bounding_boxes.remove(box)
                merged = True
        if not merged:
            merged_boxes.append(tuple(merged_box))

    # Draw merged bounding boxes on the original image
    for box in merged_boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    # Display the image with merged bounding boxes
    cv2.imshow('Image with Merged Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the actual path to your image
# Adjust the proximity_threshold as needed
join_close_and_intersecting_boxes('your_image_path.jpg', proximity_threshold=20)


