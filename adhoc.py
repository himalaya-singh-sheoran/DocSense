def minimal_augmentation(image):
    # Randomly select an augmentation
    augmentation = random.choice(['flip', 'rotate', 'scale', 'brightness', 'blur', 'contrast', 'crop', 'color_shift', 'noise', 'affine'])

    if augmentation == 'flip':
        # Flip the image horizontally
        augmented_image = cv2.flip(image, 1)  # Change the flip code (0, 1, or -1) for different directions
    elif augmentation == 'rotate':
        # Rotate the image by a small angle (-10 to 10 degrees)
        angle = random.randint(-10, 10)
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        augmented_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    elif augmentation == 'scale':
        # Scale the image by a slight factor (90% to 110%)
        scale_factor = random.uniform(0.9, 1.1)
        scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        # Ensure the scaled image matches the original size
        augmented_image = cv2.resize(scaled_image, (image.shape[1], image.shape[0]))
    elif augmentation == 'brightness':
        # Adjust brightness by adding a random value to each pixel (brighten or darken)
        brightness_value = random.randint(-20, 20)
        augmented_image = np.clip(image.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)
    elif augmentation == 'blur':
        # Apply Gaussian blur with a small kernel size (3x3 or 5x5)
        kernel_size = random.choice([3, 5])
        augmented_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif augmentation == 'contrast':
        # Adjust contrast by applying a random factor to the image
        contrast_value = random.uniform(0.8, 1.2)
        augmented_image = np.clip(image.astype(np.float32) * contrast_value, 0, 255).astype(np.uint8)
    elif augmentation == 'crop':
        # Randomly crop a region from the image
        rows, cols = image.shape[:2]
        crop_size = min(rows, cols) - 20  # Adjust the cropping size as needed
        start_row = random.randint(0, rows - crop_size)
        start_col = random.randint(0, cols - crop_size)
        augmented_image = image[start_row:start_row + crop_size, start_col:start_col + crop_size]
    elif augmentation == 'color_shift':
        # Slightly shift colors by changing the hue and saturation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_shift = random.randint(-10, 10)
        sat_shift = random.randint(-10, 10)
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + hue_shift, 0, 179)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + sat_shift, 0, 255)
        augmented_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    elif augmentation == 'noise':
        # Add subtle noise to the image
        noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
        augmented_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    elif augmentation == 'affine':
        # Apply affine transformation for slight perspective changes
        rows, cols = image.shape[:2]
        pts1 = np.float32([[10, 10], [cols - 10, 10], [10, rows - 10]])
        pts2 = np.float32([[0, 0], [cols, 0], [0, rows]])
        matrix = cv2.getAffineTransform(pts1, pts2)
        augmented_image = cv2.warpAffine(image, matrix, (cols, rows))

    return augmented_image
