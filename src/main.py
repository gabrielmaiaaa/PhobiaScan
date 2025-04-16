# Para conseguir lidar com a analise de sentimentos no rosto humano será necessário utilizar a biblioteca TensorFlow com o ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Para conseguirmos lidar com nosso dataset teremos que definir os caminhos dele
dir_train = "../data/Fer2013Dataset/test"
dir_test = "../data/Fer2013Dataset/train"

# Agora precisamos criar varaiveis que vão conter transformções que queremos aplicar em nossos datasets.
datagen_train = ImageDataGenerator(
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    horizontal_flip = True,
    rescale = 1./255,
    validation_split = 0.2
)

datagen_test = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
)

# Agora iremos carregar o caminho das imagens para poder preparar elas para treinamento em um modelo
train_generator = datagen_train.flow_from_directory(
    directory = dir_train,
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical",
    subset = "training"
)

validation_generator = datagen_train.flow_from_directory(
    directory = dir_test,
    target_size = (48, 48),
    batch_size = 64,
    color_mode = "grayscale",
    class_mode = "categorical",
    subset = "validation"
)

# Verificcando nossa base
for images, _ in train_generator:
    plt.imshow(images[0])
    plt.show()
    break

# Criando uma CNN para nossa aplicação
# Agora que já cuidamos do preprocessamento das nossa imagens, podemos utilizá-las para treinar um modelo CNN. O que iremos utilizar será o mini_XCEPTION
input_shape = (48, 48, 1)
num_classes = 7
l2_regularization = 0.01
patience = 100

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras import layers

regularization = l2(l2_regularization)

# Primeiro aplicamos um entrada com o numero de pixels (48,48) e a escala (1), grayscale.
img_input = Input(input_shape)

# Temos nossa primeira camada convolucional para nosso modelo
# A bias é False, pois aplicamos a normalização na linha seguinte `BatchNormalization`.
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
            use_bias=False)(img_input)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
            use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# Depois fazemos um loop de bloco para cada camada posterior. Está em loop, pois serão o mesmo bloco de comando e somente o valor do filtro mudará

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

# No fim, salvamos nosso modelo criado numa variável
x = Conv2D(num_classes, (3, 3),
        # kernel_regularizer=regularization,
        padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax', name='predictions')(x)

model = Model(img_input, output)

# Resumo da arquitetura de rede feita para nosso modelo
model.summary()

# Compilando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Agora iremos trainar nosso modelo com base nos preprocessamentos feitos
from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,   
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "melhor_modelo.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stop, checkpoint]
)

# Plotamos o gráfico de perda durante o treinamento
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Acurácia')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Plotamos o gráfico de acurácia durante o treinamento
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Salvamos nosso modelo
model.save("modelo_emocoes.keras")

# Dessa forma, podemos agora sempre carregar nosso modelo para testar
from keras.models import load_model
model = load_model("modelo_emocoes.keras")

# Teste
# Agora, podemos realizar testes para verificar se nosso modelo está reconhcendo ou não
my_image = plt.imread("../data/test/tes.jfif")
plt.imshow(my_image)

# Aplicamos transformações de tamanho para ficar no nosso padrão (48,48)
from skimage.transform import resize
my_image_resized = resize(my_image, (48,48,3))

img = plt.imshow(my_image_resized)

# Temos que aplicar uma transformação da imagem pro cinza, visto que treinamos um modelo com a escala cinza
import numpy as np
from skimage.color import rgb2gray

my_image_gray = rgb2gray(my_image_resized) 
my_image_gray = np.expand_dims(my_image_gray, axis=-1)  
my_image_gray = np.expand_dims(my_image_gray, axis=0)

probabilities = model.predict(my_image_gray)

plt.imshow(np.squeeze(my_image_gray))
plt.show()

# Por fim, verificamos as probabilidades de cada emoção
number_to_class = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Second most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])
print("Third most likely class:", number_to_class[index[4]], "-- Probability:", probabilities[0,index[4]])
print("Fourth most likely class:", number_to_class[index[3]], "-- Probability:", probabilities[0,index[3]])
print("Fifth most likely class:", number_to_class[index[2]], "-- Probability:", probabilities[0,index[2]])