import os
import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO

# Função que carrega os modelos que treinamos e o Yolo
def get_models():
    emotion_labels = ['angry', 'disgust', 'fear', 'neutral', 'surprise']

    current_dir = os.path.dirname(__file__)

    # Caminhos dos modelos
    model_paths = {
        'yolo': os.path.abspath(os.path.join(current_dir, '..', '..', 'models', 'yolov8n-face.pt')),
        'mini_xception': os.path.abspath(os.path.join(current_dir, '..', '..', 'modelo_emocoes.keras'))
        # 'mini_xception': os.path.abspath(os.path.join(current_dir, '..', '..', 'modelo_emocoes.keras'))
    }

    # Carregando modelos
    model_face = YOLO(model_paths['yolo'])
    model_mini_xception = load_model(model_paths['mini_xception'])

    return model_face, model_mini_xception, emotion_labels

# Função que criar as pastas e salvas as imagens dos rostos reconheicdos, com cor e gray
def save_dataset(img, img_gray, name, name_gray, img_id):
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '..', '..', 'data', name)
    
    # Define os caminhos completos
    paths = {
        'color': os.path.join(data_dir, 'Color'),
        'gray': os.path.join(data_dir, 'Gray')
    }
    
    # Cria os diretórios se não existirem
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        # print(f"Pasta '{path}' criada com sucesso!")
    
    # Gera os nomes dos arquivos
    filenames = {
        'color': f"{name}.{img_id}.jpg",
        'gray': f"{name_gray}.{img_id}.jpg"
    }
    
    # Salva as imagens
    cv2.imwrite(os.path.join(paths['color'], filenames['color']), img)
    cv2.imwrite(os.path.join(paths['gray'], filenames['gray']), img_gray)

# Função que detecta o nosso rosto e dis qual sentimento predominante
def detect(img, name, img_id, models):
    model_face, model_mini_xception, emotion_labels = models

    results = model_face(img, verbose=False, conf=0.5)

    color = {'blue': (255,0,0), 'green': (0,255,0), 'red': (0,0,255), 'white': (255,255,255)}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy() 

        for box in boxes:
            # Pega as coordenadas xy próxima dos rostos
            x1, y1, x2, y2 = map(int, box[:4])

            # Cria um box ao redor do nosso rosto
            cv2.rectangle(img, (x1, y1), (x2, y2), color['blue'], 2)
            
            # Realiza um recorte somente da área do rosto
            roi_img = img[y1:y2, x1:x2]
            
            if roi_img.size == 0:
                continue 
            
            # Aplicando o preprocessamento para noso modelo consequir analisar a imagem
            roi_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized / 255.0
            roi_reshaped = np.expand_dims(roi_normalized, axis=(0, -1))
            
            # Modelo nosso devolvendo nossa probabilidade e qual emoção é
            probabilities = model_mini_xception.predict(roi_reshaped, verbose=0)
            print(probabilities)
            predicted_class = np.argmax(probabilities)
            print(predicted_class)
            emotion = emotion_labels[predicted_class]
            print(emotion)
            confidence = np.max(probabilities) * 100
            
            # Printando na câmera a emoção e a probabilidade
            cv2.putText(img, f"Emocao: {emotion}", (x1, y1 - 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['blue'], 2)
            cv2.putText(img, f"Confianca: {confidence:.1f}%", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color['blue'], 2)
            
            # Salvando imagens para uma analise humana
            save_dataset(roi_img, roi_gray, name, name+'_Gray', img_id)

    return img


def main():
    models = get_models()

    img_id = 0

    video_capture = cv2.VideoCapture(0)

    name = input(   '--------------------------------------' \
                    '--------------------------------------' \
                    '--------------------------------------\n' \
                    'Digite seu nome: '                             )

    while True:
        _, img = video_capture.read()
        img = detect(img, name, img_id, models)
        cv2.imshow("Detector de Emocao", img)
        img_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()