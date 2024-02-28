import cv2

def enhance_image(image):
    # Denoise the image
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    # Enhance contrast
    lab = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    contrast_enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return contrast_enhanced_image

# Load the image
image = cv2.imread('input_image.jpg')

# Enhance the image
enhanced_image = enhance_image(image)

# Display the result
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
cv2.imwrite('enhanced_image.jpg', enhanced_image)
