import cv2
import flask
from flask import Response
import threading
from ultralytics import YOLO
import numpy as np
import datetime

app = flask.Flask(__name__)

CAMERA_URL = 'rtsp://admin:Jaquio@10.11.201.6:554/live/main'
model = YOLO('yolov8n.pt')
target_classes = [0]  # только люди для тепловой карты

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_URL)
        self.lock = threading.Lock()
        self.heatmap = None
        self.heatmap_start_time = datetime.datetime.now()
    
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
        
        # ПРОСТОЕ СЛОЖЕНИЕ: тепловая карта только накапливается
        self.heatmap = self.heatmap + new_heat
    
    def apply_heatmap(self, frame):
        if self.heatmap is None or np.max(self.heatmap) == 0:
            return frame
        
        # Нормализуем тепловую карту для визуализации (относительно максимума)
        if np.max(self.heatmap) > 0:
            heatmap_norm = (self.heatmap / np.max(self.heatmap) * 255).astype(np.uint8)
        else:
            heatmap_norm = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # Применяем цветовую карту
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        
        # Наложение тепловой карты на оригинальное изображение
        overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        
        # Добавляем информацию о статистике
        elapsed_time = datetime.datetime.now() - self.heatmap_start_time
        total_heat = np.sum(self.heatmap)
        
        info_text = f"Time: {elapsed_time.seconds//3600:02d}:{(elapsed_time.seconds%3600)//60:02d} | Activity: {total_heat:.0f}"
        cv2.putText(overlay, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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
        
        # Обновляем тепловую карту только когда есть обнаружения
        if len(detections) > 0:
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
    
    def get_heatmap_statistics(self):
        if self.heatmap is None:
            return None
    
        # Преобразуем сумму активности в человеко-секунды
        total_human_seconds = int(np.sum(self.heatmap))
    
        # Конвертируем в часы:минуты:секунды
        hours = total_human_seconds // 3600
        minutes = (total_human_seconds % 3600) // 60
        seconds = total_human_seconds % 60
    
        # Форматируем по-разному для разных случаев
        time_formats = {
            'detailed': f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            'short': f"{hours}ч {minutes}м",
            'very_short': f"{hours}ч" if hours > 0 else f"{minutes}м"
    }
    
        stats = {
            'total_presence': time_formats['short'],  # Самый читаемый вариант
            'total_presence_detailed': time_formats['detailed'],
            'total_activity_points': float(np.sum(self.heatmap)),  # Оригинальная метрика
            'max_activity': float(np.max(self.heatmap)),
            'collection_duration_hours': (datetime.datetime.now() - self.heatmap_start_time).total_seconds() / 3600
    }
        return stats

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
            <title>Store Heatmap Analytics</title>
            <meta charset="UTF-8">
            <style>
                body { margin: 0; padding: 0; background: black; }
                img { width: 100vw; height: 100vh; object-fit: contain; }
                .info { position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.5); padding: 10px; }
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

@app.route('/reset_heatmap')
def reset_heatmap():
    camera.heatmap = None
    camera.heatmap_start_time = datetime.datetime.now()
    return "Heatmap reset successfully"

@app.route('/statistics')
def statistics():
    stats = camera.get_heatmap_statistics()
    if stats:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Store Analytics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .stats {{ background: #f5f5f5; padding: 20px; border-radius: 10px; }}
                .reset-btn {{ background: #ff4444; color: white; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }}
            </style>
        </head>
        <body>
            <h2>Store Heatmap Statistics</h2>
            <div class="stats">
                <p><strong>Collection Time:</strong> {stats['duration_hours']:.2f} hours</p>
                <p><strong>Total Activity:</strong> {stats['total_activity']:.0f} points</p>
                <p><strong>Max Activity in Zone:</strong> {stats['max_activity']:.0f} points</p>
                <p><strong>Average Activity:</strong> {stats['average_activity']:.2f} points</p>
            </div>
            <br>
            <a href="/reset_heatmap" class="reset-btn">Reset Statistics</a>
            <br><br>
            <a href="/">Back to Camera</a>
        </body>
        </html>
        """
    return "No statistics available"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=False)
