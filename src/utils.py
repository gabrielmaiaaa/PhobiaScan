import os
import shutil
# from tqdm import tqdm 
import numpy as np

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize


# def populandoDataset():
#     fer2013 = 'data/Fer2013'
#     affect = 'data/Affectnet'

#     newFer = os.path.join('data/Fer2013New')

#     os.makedirs(newFer, exist_ok=True)

#     mapeamento_emocoes = {
#         'disgust': 'disgust',
#         'fear': 'fear',
#         'neutral': 'neutral',
#         'surprise': 'surprise'
#     }

#     os.makedirs(newFer, exist_ok=True)
#     for emocao in mapeamento_emocoes.values():
#         os.makedirs(os.path.join(newFer, emocao), exist_ok=True)

#     for diretorio, subpastas, arquivos in tqdm(os.walk(affect)):
#         current_emocao = os.path.basename(diretorio)

#         if current_emocao in mapeamento_emocoes:
#             emocao_destino = mapeamento_emocoes[current_emocao]
#             destino_dir = os.path.join(newFer, emocao_destino)

#             for arquivo in arquivos:
#                 try:
#                     img_path = os.path.join(diretorio, arquivo)
#                     image = imread(img_path)

#                     if image.ndim == 3:
#                         image = rgb2gray(image)
#                     image = resize(image, (48, 48))

#                     image = (image * 255).astype(np.uint8)

#                     new_filename = f"affect_{diretorio}"
#                     save_path = os.path.join(destino_dir, new_filename)
                    
#                     imsave(save_path, image)

#                 except Exception as e:
#                     print(f'Erro ao processar: {arquivo}, {e}')
#                     continue

def plotGraficos(final_model_filename, validation_generator, train_generator, val_acc, type, dir):
    model = load_model(final_model_filename)

    my_image_path = "../PhobiaScan/data/gab/Color/gab.153.jpg"
    my_image = imread(my_image_path)

    if my_image.ndim == 3:
        my_image_gray = rgb2gray(resize(my_image, (48, 48)))
    else:
        my_image_gray = resize(my_image, (48, 48))

    my_image_gray = np.expand_dims(my_image_gray, axis=-1) 
    my_image_gray = np.expand_dims(my_image_gray, axis=0)  
    my_image_gray = my_image_gray.astype('float32') / 255.0

    probabilities = model.predict(my_image_gray)

    number_to_class = list(train_generator.class_indices.keys())

    index = np.argsort(probabilities[0, :])[::-1]
    for i in range(min(5, len(number_to_class))):
        print(f"{i+1}ª classe mais provável: {number_to_class[index[i]]} -- Probabilidade: {probabilities[0, index[i]]:.3f}")

    y_test_pred = model.predict(validation_generator)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    y_test_true = validation_generator.classes

    number_to_class = list(validation_generator.class_indices.keys())

    print("Matriz de confusão (Teste):")
    print(confusion_matrix(y_test_true, y_test_pred))
    print("Relatório de classificação (Teste):")
    print(classification_report(y_test_true, y_test_pred, target_names=number_to_class))

    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    validation_generator.reset()
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[class_names[i] for i in range(len(class_names))],
            yticklabels=[class_names[i] for i in range(len(class_names))])
    plt.xlabel('Previsão')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão ' + type)
    plt.savefig(f'{dir}/confusion_matrix_{type}_{val_acc:.2f}.png')
    # plt.show()

def saveTxt(newDir, val_acc, val_loss, l2, dropout, batch_size, time):
    with open(f'{newDir}/data.txt', 'w') as f:
        f.write(f'----Best----\n')
        f.write(f'Valor da acurácia: {val_acc['Best']} \n')
        f.write(f'Valor da perda: {val_loss['Best']} \n')
        f.write(f'\n-----Last----\n')
        f.write(f'Valor da acurácia: {val_acc['Last']} \n')
        f.write(f'Valor da perda: {val_loss['Last']} \n')
        f.write(f'\n-----L2----\n')
        f.write(f'Valor da L2 Regularization: {l2} \n')
        f.write(f'\n-----Dropout----\n')
        f.write(f'Valor do Dropout: {dropout} \n')
        f.write(f'\n-----Batch Size----\n')
        f.write(f'Batch Size: {batch_size} \n')
        f.write(f'\n-----Time----\n')
        f.write(f'Time gasto: {time} \n')