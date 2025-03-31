import cv2
import numpy as np

def load_yolo():
    """Load YOLOv4 model and classes"""
    net = cv2.dnn.readNet("yolo_files/yolov4.weights", "yolo_files/yolov4.cfg")
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    output_layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

def detect_people(frame, net, output_layers):
    """Detect people in frame using YOLO"""
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Person class
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    people = []
    for i in range(len(boxes)):
        if i in indexes:
            people.append(boxes[i])
    return people

def process_video(input_path, output_path):
    """Process video and display real-time results"""
    net, classes, output_layers = load_yolo()
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        people = detect_people(frame, net, output_layers)
        
        for (x, y, w, h) in people:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        count = len(people)
        cv2.putText(frame, f"People: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        cv2.imshow("Live Crowd Counting", frame)
        
        if cv2.waitKey(1) == 27:  
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")

# input_video = "test2.mp4"  
# output_video = "output_counted.mp4"  
# process_video(input_video, output_video)