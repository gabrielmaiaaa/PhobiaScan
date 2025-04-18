import os
import cv2
import numpy as np
from keras.models import load_model

def save_dataset(img, name, img_id):
    base_dir = os.path.dirname(__file__) 
    save_path = os.path.join(base_dir, '..', '..', 'data', name) 
    filename = f"{name}.{img_id}.jpg"
    full_path = os.path.join(save_path, filename)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Pasta '{save_path}' criada com sucesso!")

    cv2.imwrite(full_path, img)

def draw_face_box(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []

    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        coords.append((x, y, w, h))

    return coords, img

def detect(img, faceCascade, name, img_id, model, emotion_labels):
    color = {'blue': (255,0,0), 'green': (0,255,0), 'red': (0,0,255)}
    coords, img = draw_face_box(img, faceCascade, 1.1, 10, color['blue'], "Voce")

    if len(coords) == 4:
        roi_img = img[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[2]]

        roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (48, 48))
        roi_normalized = roi_resized / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=(0, -1))

        probabilities = model.predict(roi_reshaped)
        predicted_class = np.argmax(probabilities)
        emotion = emotion_labels[predicted_class]
        confidence = np.max(probabilities) * 100

        cv2.putText(img, f"Emocao: {emotion}", (coords[0], coords[1]-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['green'], 2)
        cv2.putText(img, f"Confianca: {confidence:.1f}%", (coords[0], coords[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['green'], 2)
        
        save_dataset(roi_img, name, img_id)
        save_dataset(roi_gray, name+'_Gray', img_id)

    return img

def main():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'models', 'modelo_emocoes.keras'))

    model = load_model(model_path)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)
    img_id = 0

    name = input('Seu nome: ')

    while True:
        _, img = video_capture.read()
        img = detect(img, faceCascade, name, img_id, model, emotion_labels)
        cv2.imshow("Te vejo", img)
        img_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()