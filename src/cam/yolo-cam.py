import os
import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO

current_dir = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'notebook', 'modelo_emocoes.keras'))
model_emotion = load_model(model_path)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


current_dir = os.path.dirname(__file__)
yolo_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'notebook', 'yolov8n-face.pt'))
model_face = YOLO(yolo_path)  
video_capture = cv2.VideoCapture(0)
img_id = 0

name = input('Seu nome: ')

def save_dataset(img, name, id, img_id):
    base_dir = os.path.dirname(__file__) 
    save_path = os.path.join(base_dir, '..', '..', 'data', name) 
    filename = f"{name}.{id}.{img_id}.jpg"
    full_path = os.path.join(save_path, filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Pasta '{save_path}' criada com sucesso!")

    cv2.imwrite(full_path, img)

def detect(img, name, img_id):
    results = model_face(img, verbose=False, conf=0.5)
    color = {'blue': (255,0,0), 'green': (0,255,0), 'red': (0,0,255), 'white': (255,255,255)}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), color['white'], 2)
            
            roi_img = img[y1:y2, x1:x2]
            
            if roi_img.size == 0:
                continue 

            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.expand_dims(roi_normalized, axis=(0, -1))
            
            probabilities = model_emotion.predict(roi_reshaped, verbose=0)
            predicted_class = np.argmax(probabilities)
            emotion = emotion_labels[predicted_class]
            confidence = np.max(probabilities) * 100
            
            cv2.putText(img, f"Emocao: {emotion}", (x1, y1 - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['blue'], 2)
            cv2.putText(img, f"Confianca: {confidence:.1f}%", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['blue'], 2)
            
            save_dataset(roi_img, name, 1, img_id)
            save_dataset(roi_gray, name + '_Gray', 1, img_id)

    return img

while True:
    _, img = video_capture.read()
    img = detect(img, name, img_id)
    cv2.imshow("Te vejo", img)
    img_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()