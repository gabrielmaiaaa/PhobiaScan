import os
import cv2
import time
import win32gui
import win32con
import pyautogui
import numpy as np
import collections
from ultralytics import YOLO
from turtle import screensize
from keras.models import load_model

def get_models():
    emotion_labels = ['disgust', 'fear', 'neutral', 'surprise']
    # emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad','surprise']

    model_paths = {
        'yolo': 'models/cam/yolov8n-face.pt',
        'mini_xceptionEmotion4': 'models/tests/RAF-DB/RAF-DB_38/Best_255_loss_0.1820_acc_0.96.keras',
        'mini_xceptionEmotion7': 'models/tests/RAF-DB2/RAF-DB2_0/Best_123_loss_0.4437_acc_0.87.keras'
    }

    model_face = YOLO(model_paths['yolo'])
    model_mini_xception = load_model(model_paths['mini_xceptionEmotion4'])

    return model_face, model_mini_xception, emotion_labels

def save_dataset(img, img_gray, name, name_gray, emotion):
    dirColor = f'data/{name}/Color/{emotion}'
    dirGray = f'data/{name}/Gray/{emotion}'

    os.makedirs(dirColor, exist_ok=True)
    arquivos = [f for f in os.listdir(dirColor) if os.path.isfile(os.path.join(dirColor, f))]
    img_id = len(arquivos) + 1
    print(img_id)
    
    paths = {
        'color': dirColor,
        'gray': dirGray
    }
    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    filenames = {
        'color': f"{name}.{img_id}.jpg",
        'gray': f"{name_gray}.{img_id}.jpg"
    }
    
    cv2.imwrite(os.path.join(paths['color'], filenames['color']), img)
    cv2.imwrite(os.path.join(paths['gray'], filenames['gray']), img_gray)

def detect(img, name, models):
    model_face, model_mini_xception, emotion_labels = models

    emotion = ''

    results = model_face(img, verbose=False, conf=0.5)

    color = {'blue': (255,0,0), 'green': (0,255,0), 'red': (0,0,255), 'white': (255,255,255)}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            # x1 -= 50
            # y1 -= 50
            # x2 += 50
            # y2 += 50
            print(x1,y1,x2,y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), color['blue'], 2)
            
            roi_img = img[y1:y2, x1:x2]
            
            if roi_img.size == 0:
                continue 
            
            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (75, 75))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.expand_dims(roi_normalized, axis=(0, -1))
            
            probabilities = model_mini_xception.predict(roi_reshaped)
            predicted_class = np.argmax(probabilities)
            emotion = emotion_labels[predicted_class]
            confidence = np.max(probabilities) * 100
            
            cv2.putText(img, f"Emocao: {emotion}", (x1, y1 - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['blue'], 2)
            cv2.putText(img, f"Confianca: {confidence:.1f}%", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['blue'], 2)
            
            save_dataset(roi_img, roi_gray, name, name+'_Gray', emotion)

    return img, emotion

def myScreenshot(numberFeedback, verificacao, fps, frameBuffer, screenSize, emotion, dir):
    tela = pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(tela), cv2.COLOR_RGB2BGR)

    frameBuffer.append(frame)

    if emotion == 'fear':        
        verificacao[0] += 1

    if verificacao[0] > 10:
        numberFeedback[0] += 1

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video = cv2.VideoWriter(f'{dir}/feedback{numberFeedback[0]}.avi', fourcc, fps, screenSize)

        for frameVideo in frameBuffer:
            video.write(frameVideo)
            
        video.release()
        frameBuffer.clear()
        verificacao[0] = 0

def main():
    models = get_models()

    name = 'Gabriel3'
    
    dir = f'Feedbacks/{name}'
    os.makedirs(dir, exist_ok=True)
    arquivos = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    numberFeedback = [len(arquivos)]
    verificacao = [0]
    fps = 5
    duracao = 30
    bufferSize = fps * duracao
    frameBuffer = collections.deque(maxlen=bufferSize)

    video_capture = cv2.VideoCapture(0)
    screenSize = tuple(pyautogui.size())

    window_name = "Detector de Emocao"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    inicialTime = time.time()
    fpsList = []

    while True:
        _, img = video_capture.read()
        img, emotion = detect(img, name, models)
        cv2.imshow(window_name, img)

        hwnd = win32gui.FindWindow(None, window_name)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        myScreenshot(numberFeedback, verificacao, fps, frameBuffer, screenSize, emotion, dir)

        endTime = time.time()
        dt = endTime - inicialTime
        inicialTime = endTime
        if dt > 0:
            fpsInstant = 1 / dt
            fpsList.append(fpsInstant)
            if len(fpsList) > 30:
                fpsList.pop(0)
            fps = sum(fpsList) / len(fpsList)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()