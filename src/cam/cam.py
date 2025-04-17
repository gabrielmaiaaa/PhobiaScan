import os
import cv2

def save_dataset(img, name, id, img_id):
    base_dir = os.path.dirname(__file__) 
    save_path = os.path.join(base_dir, '..', '..', 'data', name) 
    filename = f"{name}.{id}.{img_id}.jpg"
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
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords, img

def detect(img, faceCascade, name, img_id):
    color = {'blue': (255,0,0), 'green': (0,255,0), 'red': (0,0,255)}
    coords, img = draw_face_box(img, faceCascade, 1.1, 10, color['blue'], "Você")

    if len(coords) == 4:
        roi_img = img[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[2]]

        save_dataset(roi_img, name, 1, img_id)

    return img

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
img_id = 0

name = input('Seu nome: ')

while True:
    _, img = video_capture.read()
    img = detect(img, faceCascade, name, img_id)
    cv2.imshow("Te vejo", img)
    img_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()