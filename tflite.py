import cv2
import numpy as np
import tensorflow as tf
import os

# === CONFIG ===
MODEL_PATH = 'assets/best_float32.tflite'
LABEL_PATH = 'assets/labels.txt'
IMAGE_PATH = 'assets/images/helmet.jpg'
OUTPUT_PATH = 'assets/images/helmet_output1.jpg'
INPUT_SIZE = (640, 640)  # Changed to match model expectation (width, height)
CONFIDENCE_THRESHOLD = 0.1

# === Load Labels ===
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# === Preprocess Image ===
def preprocess_image(image_path, input_size, expected_dtype):
    image = cv2.imread(image_path)
    original = image.copy()
    image = cv2.resize(image, input_size)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if expected_dtype == np.float32:
        input_data = np.expand_dims(image_rgb, axis=0).astype(np.float32) / 255.0
    else:
        input_data = np.expand_dims(image_rgb, axis=0).astype(np.uint8)

    return input_data, original

# === Decode YOLO Output ===
def decode_predictions(feature_maps, input_size):
    boxes, class_ids, scores = [], [], []

    for output in feature_maps:
        output = np.squeeze(output)  # Shape should be [7, 8400]
        
        # Handle the output format: [class_count + 4, detection_count]
        # Transpose to get [detection_count, class_count + 4]
        if len(output.shape) == 2:
            output = output.transpose()  # Now [8400, 7]
        
        num_detections, num_values = output.shape
        
        for i in range(num_detections):
            data = output[i]
            
            # For YOLOv8-style output: [cx, cy, w, h, class_scores...]
            # Extract box coordinates
            cx, cy, w, h = data[0], data[1], data[2], data[3]
            
            # Extract class scores (remaining elements)
            class_scores = data[4:]
            
            # Get the class with highest score
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence > CONFIDENCE_THRESHOLD:
                # Convert from center format to corner format
                # Normalize coordinates
                xmin = (cx - w / 2) / input_size[0]
                ymin = (cy - h / 2) / input_size[1]
                xmax = (cx + w / 2) / input_size[0]
                ymax = (cy + h / 2) / input_size[1]

                boxes.append([ymin, xmin, ymax, xmax])
                class_ids.append(class_id)
                scores.append(confidence)

    return np.array(boxes), np.array(class_ids), np.array(scores)

# === Draw Bounding Boxes ===
def draw_boxes(image, boxes, classes, scores):
    h, w, _ = image.shape
    for i in range(len(scores)):
        ymin, xmin, ymax, xmax = boxes[i]
        class_id = int(classes[i])
        if class_id >= len(labels):  # Skip if class ID is invalid
            continue

        (left, top, right, bottom) = (
            int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
        )
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{labels[class_id]}: {scores[i]:.2f}"
        cv2.putText(
            image, label, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )
    return image

# === Run Inference ===
def run_inference():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Input shape:", input_details[0]['shape'], input_details[0]['dtype'])
    for i, d in enumerate(output_details):
        print(f"Output {i}: shape={d['shape']}, dtype={d['dtype']}")

    expected_dtype = input_details[0]['dtype']
    
    # Use the correct input size (no need to reverse)
    input_data, original_image = preprocess_image(IMAGE_PATH, INPUT_SIZE, expected_dtype)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract feature maps
    feature_maps = [interpreter.get_tensor(d['index']) for d in output_details]
    boxes, class_ids, scores = decode_predictions(feature_maps, INPUT_SIZE)

    # Draw results
    result_image = draw_boxes(original_image, boxes, class_ids, scores)

    # === Print detection summary ===
    class_counts = {}
    for class_id in class_ids:
        if class_id >= len(labels):
            continue
        label = labels[class_id]
        class_counts[label] = class_counts.get(label, 0) + 1

    print("\n🟩 Detected Objects:")
    if not class_counts:
        print("No objects detected above the confidence threshold.")
    else:
        for label, count in class_counts.items():
            print(f"- {label}: {count}")

    # Save image
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, result_image)
    print(f"\nSaved result to: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_inference()