import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D
from keras.regularizers import l2
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

current_dir = os.path.dirname(__file__)

# Caminhos dos modelos
model_paths = {
    'yolo': os.path.abspath(os.path.join(current_dir, '..', '..', 'PhobiaScan', 'data', 'images2', 'FER2013Train')),
    'mini_xception': os.path.abspath(os.path.join(current_dir, '..', '..', 'PhobiaScan', 'src', 'fer2013plus.csv')),
    'yolo1': os.path.abspath(os.path.join(current_dir, '..', '..', 'PhobiaScan', 'data', 'images2', 'FER2013Test'))
}


# Carregar o arquivo CSV
df = pd.read_csv(model_paths['mini_xception'])  # Substitua pelo caminho do seu arquivo CSV

# Definir as colunas de emoções
emotion_columns = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# Filtrar apenas as colunas necessárias e linhas com NF=0 (supondo que NF significa Not Face)
df = df[df['NF'] == 0]

# Função para converter votos em probabilidades (soft labels)
def votes_to_probabilities(row):
    votes = row[emotion_columns].values
    total_votes = np.sum(votes)
    if total_votes > 0:
        return votes / total_votes
    else:
        return np.zeros(len(emotion_columns))  # Caso não haja votos

# Aplicar a conversão para cada linha
y = df.apply(votes_to_probabilities, axis=1)
y = np.array(y.tolist(), dtype='float32')

# Carregar as imagens
def load_images_from_csv(df, image_dir):
    images = []
    for idx, row in df.iterrows():
        img_path = f"{image_dir}/{row['filename']}"
        try:
            img = plt.imread(img_path)
            if len(img.shape) == 3:  # Se for colorida, converter para grayscale
                img = np.mean(img, axis=2)
            img = resize(img, (48, 48))  # Redimensionar para 48x48
            images.append(img)
        except Exception as e:
            print(f"Erro ao carregar imagem {img_path}: {e}")
            # Adicionar uma imagem vazia para manter o alinhamento com os rótulos
            images.append(np.zeros((48, 48)))
            continue
    
    # Converter para numpy array e garantir o tipo float32
    return np.array(images, dtype='float32')

# Carregar as imagens (separando treino e validação)
train_df = df[df['usage'] == 'Training']
val_df = df[df['usage'] == 'PublicTest']  # Ou 'Validation' dependendo do seu CSV

X_train = load_images_from_csv(train_df, model_paths['yolo'])
y_train = y[train_df.index]

X_val = load_images_from_csv(val_df, model_paths['yolo1'])
y_val = y[val_df.index]

# Adicionar dimensão do canal (para grayscale) e normalizar
X_train = np.expand_dims(X_train, axis=-1) / 255.0
X_val = np.expand_dims(X_val, axis=-1) / 255.0

# Criar geradores de dados com aumento para treino
datagen_train = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

datagen_val = ImageDataGenerator()

# Configurar os geradores
batch_size = 64

train_generator = datagen_train.flow(
    X_train, y_train,
    batch_size=batch_size,
    shuffle=True
)

validation_generator = datagen_val.flow(
    X_val, y_val,
    batch_size=batch_size,
    shuffle=False
)

# Verificar uma amostra
sample_img, sample_label = next(train_generator)
plt.imshow(np.squeeze(sample_img[0]), cmap='gray')
plt.title(f"Label: {sample_label[0]}")
plt.show()

# Definir a arquitetura do modelo (igual ao seu original)
input_shape = (48, 48, 1)
num_classes = 7
l2_regularization = 0.01
patience = 100

regularization = l2(l2_regularization)

img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
            use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
            use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

filter_list = [16, 32, 64, 128]
for n in filter_list:
    residual = Conv2D(n, (1, 1), strides=(2, 2),
                        padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(n, (3, 3), padding='same',
                    depthwise_regularizer=regularization,
                    pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(n, (3, 3), padding='same',
                    depthwise_regularizer=regularization,
                    pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax', name='predictions')(x)

model = Model(img_input, output)

# Compilar o modelo - usando categorical_crossentropy que funciona bem com soft labels
model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,   
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "melhor_modelo_votos.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("Verificando tipos e shapes:")
print(f"X_train dtype: {X_train.dtype}, shape: {X_train.shape}")
print(f"y_train dtype: {y_train.dtype}, shape: {y_train.shape}")
print(f"X_val dtype: {X_val.dtype}, shape: {X_val.shape}")
print(f"y_val dtype: {y_val.dtype}, shape: {y_val.shape}")

# Verificar se há NaNs
print(f"NaNs em X_train: {np.isnan(X_train).sum()}")
print(f"NaNs em y_train: {np.isnan(y_train).sum()}")

# Treinar o modelo
hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stop, checkpoint]
)

# Plotar resultados
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss do Modelo')
plt.ylabel('Loss')
plt.xlabel('Época')
plt.legend(['Treino', 'Validação'], loc='upper right')

plt.tight_layout()
plt.show()

# Salvar o modelo
model.save("modelo_emocoes_votos.keras")

# Função para mostrar a previsão em uma nova imagem
def predict_emotion(model, image_path):
    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title("Imagem Original")
    plt.show()
    
    # Pré-processamento
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    img = resize(img, (48, 48))
    img_gray = np.expand_dims(img, axis=-1)  # Adicionar dimensão do canal
    img_gray = np.expand_dims(img_gray, axis=0) / 255.0  # Adicionar dimensão do batch e normalizar
    
    # Previsão
    probabilities = model.predict(img_gray)
    
    # Mostrar resultados
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    sorted_indices = np.argsort(probabilities[0])[::-1]
    
    print("Previsões de emoção:")
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {emotion_labels[idx]}: {probabilities[0][idx]*100:.2f}%")
    
    plt.imshow(np.squeeze(img_gray[0]), cmap='gray')
    plt.title(f"Principal emoção: {emotion_labels[sorted_indices[0]]}")
    plt.show()

# Testar com uma imagem
predict_emotion(model, "../data/gab/Color/gab.153.jpg")