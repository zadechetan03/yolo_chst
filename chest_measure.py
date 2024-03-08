import cv2
import numpy as np

def measure_chest_size(image_path, reference_height_cm):
    # Load YOLOv3 model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    
    # Load input image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Preprocess image
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    # print(layer_names)
    # print(net.getUnconnectedOutLayersNames())
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass to get output of output layers
    outputs = net.forward(output_layers)
    
    # Initialize variables for measurements
    max_human_area = 0
    chest_size_px = 0
    
    # Process detections
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            
            if class_id == 0:  # Class ID 0 corresponds to 'person'
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Assuming chest is approximately at the center of the body
                    x = max(0, center_x - w // 2)
                    y = max(0, center_y - h // 2)
                    
                    # Calculate area of the bounding box
                    area = w * h
                    
                    # Select the largest detected human body
                    if area > max_human_area:
                        max_human_area = area
                        chest_size_px = w  # Use height of bounding box as chest size
    
    # Convert reference height from cm to pixels (assuming 1 cm = 10 pixels)
    reference_height_px = reference_height_cm * 10
    
    # Calculate chest size in cm
    chest_size_cm = (chest_size_px / reference_height_px) * reference_height_cm
    
    # Draw bounding box around detected human body
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Annotate chest size
    text = f"Chest Size: {chest_size_cm:.2f} cm"
    cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the annotated image
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return chest_size_cm

# Path to the input image
image_path = "saif1.jpeg"

# Reference height of the body in cm
reference_height_cm = 160  # Adjust this value based on your reference

# Measure chest size
chest_size_cm = measure_chest_size(image_path, reference_height_cm)

# Print the measured chest size
print("Measured Chest Size:", chest_size_cm, "cm")