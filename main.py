import os
import cv2
import numpy as np
import tensorflow as tf

# -----------------------------
# File paths
# -----------------------------
MODEL_PATH = "rice_model.keras"
CLASSES_PATH = "classes.txt"

# -----------------------------
# Check files
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("rice_model.keras not found. Please run train.py first.")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError("classes.txt not found. Please run train.py first.")

# -----------------------------
# Load trained model and classes
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print("Model loaded successfully.")
print("Classes:", class_names)

# -----------------------------
# Start webcam
# 0 = default laptop webcam
# 1 = external webcam
# -----------------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing 0 to 1.")

# Set camera size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

IMG_SIZE = 224

print("Webcam started.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame from camera.")
        break

    # Copy frame for display
    display_frame = frame.copy()

    # Convert BGR to RGB because TensorFlow uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize image for model
    img = cv2.resize(rgb_frame, (IMG_SIZE, IMG_SIZE))

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    # Text color
    if confidence >= 80:
        color = (0, 255, 0)      # Green
    elif confidence >= 50:
        color = (0, 255, 255)    # Yellow
    else:
        color = (0, 0, 255)      # Red

    label = f"{predicted_class}: {confidence:.2f}%"

    # Draw background rectangle
    cv2.rectangle(display_frame, (20, 20), (650, 90), (0, 0, 0), -1)

    # Draw prediction text
    cv2.putText(
        display_frame,
        label,
        (35, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    # Show all class confidence
    y = 130
    for i, class_name in enumerate(class_names):
        text = f"{class_name}: {predictions[i] * 100:.2f}%"
        cv2.putText(
            display_frame,
            text,
            (35, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 35

    # Show camera window
    cv2.imshow("Real-time Rice Leaf Disease Detection", display_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release camera
cap.release()
cv2.destroyAllWindows()