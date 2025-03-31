import cv2
import numpy as np

# Load YOLOv4
net = cv2.dnn.readNet("yolo_files/yolov4.weights", "yolo_files/yolov4.cfg")
classes = []
with open("yolo_files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or "video.mp4" for a file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Detect objects (YOLO expects 416x416)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class 0 = 'person' in COCO
                # Object detected is a person
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-max suppression to avoid duplicate boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes and count people
    people_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            people_count += 1

    # Display count
    cv2.putText(frame, f"People: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Crowd Counter", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()