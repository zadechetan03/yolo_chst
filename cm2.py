import cv2
import numpy as np

# Function to detect chest width
def detect_chest(image_path, reference_height_cm):
    # Load input image
    image = cv2.imread(image_path)

    # Run OpenPose on the input image to obtain key points
    # Use your OpenPose installation and configuration
    # Store the key points in a suitable data structure

    # Assuming key points are stored in the format [x, y] for each key point
    # For demonstration purposes, we randomly generate key points representing the shoulders
    left_shoulder = [300, 500]  # Example key point for left shoulder
    right_shoulder = [600, 500]  # Example key point for right shoulder

    # Calculate the distance between the shoulders to get the chest width
    chest_width_pixels = abs(right_shoulder[0] - left_shoulder[0])

    # Convert reference height from cm to pixels (assuming 1 cm = 10 pixels)
    reference_height_px = reference_height_cm * 10
    
    # Calculate the scaling factor for width
    scaling_factor = reference_height_px / 170

    # Convert chest width from pixels to centimeters
    chest_width_cm = chest_width_pixels / scaling_factor

    # Annotate the input image with the detected chest width
    cv2.line(image, (left_shoulder[0], left_shoulder[1]), (right_shoulder[0], right_shoulder[1]), (0, 255, 0), 2)
    cv2.putText(image, f'Chest Width: {chest_width_cm:.2f} cm', (left_shoulder[0], left_shoulder[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated image
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return chest_width_cm

# Path to the input image
image_path = "saif1.jpg"

# Reference height of the body in cm
reference_height_cm = 170  # Adjust this value based on your reference

# Detect chest width and display annotated image
chest_width_cm = detect_chest(image_path, reference_height_cm)

# Print the measured chest width
print("Measured Chest Width:", chest_width_cm, "cm")
