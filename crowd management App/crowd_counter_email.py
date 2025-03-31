import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

from crow_counter_video import detect_people, load_yolo

SMTP_SERVER = "smtp.gmail.com"  
SMTP_PORT = 587
EMAIL_ADDRESS = "adhikaryswapnanil@gmail.com"
EMAIL_PASSWORD = "apou meaq ysqy txtx " 
RECIPIENT_EMAIL = "shekharadhikary024@gmail.com"

last_alert_time = 0
ALERT_COOLDOWN = 30  
def send_alert_email(count):
    """Send email alert when crowd exceeds threshold"""
    global last_alert_time
    
    current_time = time.time()
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return  
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"🚨 Crowd Alert: {count} people detected!"
        
        body = f"""
        <h2>Crowd Limit Exceeded!</h2>
        <p>Current crowd count: <strong>{count}</strong> people</p>
        <p>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        msg.attach(MIMEText(body, 'html'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"Alert email sent! People count: {count}")
        last_alert_time = current_time
        
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def process_video(input_path, output_path):
    """Main processing function with email alerts"""
    net, classes, output_layers = load_yolo()
    cap = cv2.VideoCapture(input_path)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 
                         int(cap.get(cv2.CAP_PROP_FPS)),
                         (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        people = detect_people(frame, net, output_layers)
        count = len(people)
        
        # Visualization
        for (x, y, w, h) in people:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        cv2.putText(frame, f"People: {count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Alert system
        if count > 10:
            cv2.putText(frame, "ALERT: Crowded!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            send_alert_email(count)
        
        out.write(frame)
        cv2.imshow("Crowd Monitoring", frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    process_video("test2.mp4", "output.mp4")