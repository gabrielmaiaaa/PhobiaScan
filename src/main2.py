from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D, Dropout
from keras.models import Model, load_model
from keras.regularizers import l2
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize
from skimage.color import rgb2gray
import os

# Diretórios
dir_train = "../PhobiaScan/data/Fer2013/train"
dir_test = "../PhobiaScan/data/Fer2013/test"

dir_train_plus = "../PhobiaScan/data/Fer2013Plus/FER2013Train"
dir_valid_plus = "../PhobiaScan/data/Fer2013Plus/FER2013Valid"
dir_test_plus = "../PhobiaScan/data/Fer2013Plus/FER2013Test"

# Data augmentation para treino e validação
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,  
    zoom_range=0.15,   
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  
    rescale=1./255,
    validation_split=0.2  # Split interno para validação
)
datagen_test = ImageDataGenerator(
    rescale=1./255
)

# Gerador de treino (80% do dir_train)
train_generator = datagen_train.flow_from_directory(
    directory=dir_train,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    shuffle=True
)
# Gerador de validação (20% do dir_train)
validation_generator = datagen_train.flow_from_directory(
    directory=dir_train,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
test_generator = datagen_test.flow_from_directory(
    directory=dir_test,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    shuffle=False
)

# Visualização de um batch
for images, labels in train_generator:
    plt.imshow(images[0].squeeze(), cmap='gray')
    plt.title(f"Classe: {np.argmax(labels[0])}")
    plt.show()
    break

# Parâmetros do modelo
input_shape = (48, 48, 1)
num_classes = len(train_generator.class_indices)
l2_regularization = 0.01
patience = 100  
verbose = 1

# Arquitetura do modelo
regularization = l2(l2_regularization)
img_input = Input(input_shape)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.25)(x)

filter_list = [16, 32, 64, 128]
for n in filter_list:
    residual = Conv2D(n, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(n, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = SeparableConv2D(n, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

x = Conv2D(num_classes, (3, 3), padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax', name='predictions')(x)
model = Model(img_input, output)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

# Nome do melhor modelo com timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_filename = f"melhor_modelo_{timestamp}.keras"

checkpoint = ModelCheckpoint(
    checkpoint_filename,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=verbose
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Class weights automáticos
y_train = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))

# Treinamento
hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1000,
    verbose=verbose,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Plot das curvas de loss e accuracy
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Acurácia')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Salvar modelo final com timestamp
final_model_filename = f"modelo_emocoes_{timestamp}.keras"
model.save(final_model_filename)
print(f"Modelo salvo como {final_model_filename}")

# Carregar melhor modelo salvo
model = load_model(checkpoint_filename)

# Inferência em uma imagem de teste
from skimage.io import imread

# Troque pelo caminho de uma imagem real do seu dataset
my_image_path = "../PhobiaScan/data/gab/Color/gab.153.jpg"
my_image = imread(my_image_path)

# Se a imagem for colorida, converte para cinza
if my_image.ndim == 3:
    my_image_gray = rgb2gray(resize(my_image, (48, 48)))
else:
    my_image_gray = resize(my_image, (48, 48))

my_image_gray = np.expand_dims(my_image_gray, axis=-1)  # (48,48,1)
my_image_gray = np.expand_dims(my_image_gray, axis=0)   # (1,48,48,1)
my_image_gray = my_image_gray.astype('float32') / 255.0

probabilities = model.predict(my_image_gray)
plt.imshow(my_image_gray.squeeze(), cmap='gray')
plt.show()

# Classes dinâmicas
number_to_class = list(train_generator.class_indices.keys())
# Ordena por probabilidade
index = np.argsort(probabilities[0, :])[::-1]
for i in range(min(5, len(number_to_class))):
    print(f"{i+1}ª classe mais provável: {number_to_class[index[i]]} -- Probabilidade: {probabilities[0, index[i]]:.3f}")

# Avaliação no conjunto de teste
y_test_pred = model.predict(test_generator)
y_test_pred = np.argmax(y_test_pred, axis=1)
y_test_true = test_generator.classes

number_to_class = list(test_generator.class_indices.keys())

print("Matriz de confusão (Teste):")
print(confusion_matrix(y_test_true, y_test_pred))
print("Relatório de classificação (Teste):")
print(classification_report(y_test_true, y_test_pred, target_names=number_to_class))