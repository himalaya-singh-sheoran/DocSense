import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def extract_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection to extract edge features
    edges = cv2.Canny(gray, 100, 200)

    # Convert the edge image to binary
    binary_edges = np.uint8(edges > 0)

    # Reshape the binary edges into feature vectors
    features = np.argwhere(binary_edges)

    return features

def text_region_detection(image):
    # Extract features from the image
    features = extract_features(image)

    # Perform Hierarchical Agglomerative Clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=10, linkage='ward').fit(features)

    # Filter out clusters representing text regions
    text_regions = []
    for cluster_label in np.unique(clustering.labels_):
        cluster_points = features[clustering.labels_ == cluster_label]
        # Filter clusters based on size, aspect ratio, etc.
        if len(cluster_points) > 50:  # Adjust size threshold as needed
            text_regions.append(cluster_points)

    # Generate a mask for text regions
    text_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    for region in text_regions:
        for point in region:
            text_mask[point[0], point[1]] = 255

    return text_mask

# Load the image
image = cv2.imread('input_image.jpg')

# Detect text regions in the image
text_mask = text_region_detection(image)

# Display the mask
cv2.imshow('Text Mask', text_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the mask
cv2.imwrite('text_mask.jpg', text_mask)
