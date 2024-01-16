
import cv2
import numpy as np

# Load the pre-trained EAST text detector
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

def detect_text_regions(image_path):
    # Read the image
    image = cv2.imread(image_path)
    original_height, original_width = image.shape[:2]

    # Prepare the image for the EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (original_width, original_height), (123.68, 116.78, 103.94), True, False)
    net.setInput(blob)

    # Get the output layer names
    output_layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Forward pass to get the output layers
    scores, geometry = net.forward(output_layer_names)

    # Decode the bounding box coordinates
    rectangles, confidences = cv2.dnn.NMSBoxesRotated(
        boxes=np.array([cv2.boundingRect(cv2.findNonZero(score > 0.5)) for score in scores]),
        scores=confidences,
        score_threshold=0.5,
        nms_threshold=0.4
    )

    # Draw bounding boxes on the original image
    for rect in rectangles:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_image_path.jpg' with the actual path to your image
detect_text_regions('your_image_path.jpg')
