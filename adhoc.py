import cv2

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Get the rotated bounding box and resize the image
    cos_theta = abs(rotation_matrix[0, 0])
    sin_theta = abs(rotation_matrix[0, 1])
    new_width = int(height * sin_theta + width * cos_theta)
    new_height = int(height * cos_theta + width * sin_theta)
    rotated_bounding_box = cv2.transform(np.array([[0, 0], [width, 0], [0, height], [width, height]]).reshape(-1, 1, 2), rotation_matrix)
    min_x, min_y = np.min(rotated_bounding_box, axis=0).ravel()
    max_x, max_y = np.max(rotated_bounding_box, axis=0).ravel()
    translated_rotation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    rotated_image = cv2.warpPerspective(rotated_image, translated_rotation_matrix, (max_x - min_x, max_y - min_y))

    return rotated_image

# Load the image
image = cv2.imread('input_image.jpg')

# Specify the rotation angle (in degrees, clockwise)
angle = 45

# Rotate the image
rotated_image = rotate_image(image, angle)

# Display the original and rotated images
cv2.imshow('Original Image', image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

