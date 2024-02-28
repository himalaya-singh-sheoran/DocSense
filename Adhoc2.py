import cv2
import numpy as np
import random

def replace_text_regions(image, text_regions):
    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # Specify the output layers
    layer_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # Loop through detected text regions
    for region in text_regions:
        x, y, w, h = region
        # Extract the text from the original image region
        roi = image[y:y+h, x:x+w]

        # Resize region to a fixed size (320x320) required by EAST text detector
        blob = cv2.dnn.blobFromImage(roi, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)

        # Forward pass through the network to get the text detection scores and geometry
        net.setInput(blob)
        scores, geometry = net.forward(layer_names)

        # Decode the text regions from the scores and geometry
        rects, confidences = decode_predictions(scores, geometry)

        # Loop through the detected text regions
        for rect, confidence in zip(rects, confidences):
            x1, y1, x2, y2 = rect
            text = "New Text"  # Replace with the desired new text

            # Get the color of the text region
            text_color = image[y, x].tolist()

            # Overlay new text onto the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = text_color
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = x + (x2 - x1 - text_width) // 2
            text_y = y + (y2 - y1 + text_height) // 2
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return image

def decode_predictions(scores, geometry):
    # Extract the number of rows and columns from the scores
    num_rows, num_cols = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the number of rows
    for y in range(0, num_rows):
        # Extract the scores (probabilities) and geometrical data (bounding box) for each row
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles_data = geometry[0, 4, y]

        # Loop over the number of columns
        for x in range(0, num_cols):
            # If the score is not sufficiently high, skip this region
            if scores_data[x] < 0.5:
                continue

            # Calculate the offset factor as some of the geometry data is offset
            offset_factor_x = x * 4.0
            offset_factor_y = y * 4.0

            # Extract the angle for this region and calculate its sine and cosine
            angle = angles_data[x]
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            # Calculate the bounding box offsets
            offset_x = x_data0[x] + x_data2[x]
            offset_y = x_data1[x] + x_data3[x]

            # Calculate the bounding box coordinates
            x1 = int(offset_factor_x + (cos_angle * x_data0[x]) + (sin_angle * x_data2[x]))
            y1 = int(offset_factor_y - (sin_angle * x_data0[x]) + (cos_angle * x_data2[x]))
            x2 = int(offset_factor_x + (cos_angle * x_data1[x]) + (sin_angle * x_data3[x]))
            y2 = int(offset_factor_y - (sin_angle * x_data1[x]) + (cos_angle * x_data3[x]))

            # Append the bounding box coordinates and confidence to the respective lists
            rects.append((x1, y1, x2, y2))
            confidences.append(scores_data[x])

    return rects, confidences

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
