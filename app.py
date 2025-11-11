import cv2
import flask
from flask import Response
import threading
from ultralytics import YOLO
import numpy as np

app = flask.Flask(__name__)

CAMERA_URL = 'rtsp://admin:Jaquio@192.168.105.8:554/live/main'
model = YOLO('yolov8n.pt')
target_classes = [0]  # только люди для тепловой карты

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_URL)
        self.lock = threading.Lock()
        self.heatmap = None
        self.alpha = 0.1  # коэффициент затухания тепловой карты
    
    def update_heatmap(self, frame, detections):
        if self.heatmap is None:
            self.heatmap = np.zeros(frame.shape[:2], dtype=np.float32)
        
        # Создаем маску для новых обнаружений
        new_heat = np.zeros(frame.shape[:2], dtype=np.float32)
        
        for box in detections:
            x1, y1, x2, y2 = map(int, box)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Добавляем гауссово размытие вокруг центра обнаружения
            cv2.circle(new_heat, (center_x, center_y), 30, 1, -1)
        
        # Обновляем тепловую карту с затуханием
        self.heatmap = self.heatmap * self.alpha + new_heat
        self.heatmap = np.clip(self.heatmap, 0, 1)
    
    def apply_heatmap(self, frame):
        if self.heatmap is None:
            return frame
        
        # Нормализуем тепловую карту
        heatmap_norm = (self.heatmap * 255).astype(np.uint8)
        
        # Применяем цветовую карту
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Наложение тепловой карты на оригинальное изображение
        overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        return overlay
    
    def detect_objects(self, frame):
        results = model.track(frame, persist=True, verbose=False)
        detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, cls, conf, id in zip(boxes, classes, confidences, ids):
                if cls in target_classes and conf > 0.5:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # Сохраняем обнаружения людей для тепловой карты
                    detections.append([x1, y1, x2, y2])
                    
                    # Рисуем bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label = f"ID:{int(id)} {model.names[int(cls)]}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Обновляем тепловую карту
        if detections:
            self.update_heatmap(frame, detections)
        
        # Применяем тепловую карту к кадру
        frame_with_heatmap = self.apply_heatmap(frame)
        
        return frame_with_heatmap
    
    def get_frame(self):
        with self.lock:
            success, frame = self.cap.read()
            if not success:
                self.cap.release()
                self.cap = cv2.VideoCapture(CAMERA_URL)
                return None
            
            frame = self.detect_objects(frame)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

camera = Camera()

def generate_frames():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>Камера с тепловой картой</title>
            <style>
                body { margin: 0; padding: 0; background: black; }
                img { width: 100vw; height: 100vh; object-fit: contain; }
            </style>
        </head>
        <body>
            <img src="/video_feed">
        </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
