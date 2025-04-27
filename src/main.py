from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dir_train = "../PhobiaScan/data/Fer2013Dataset/train"
dir_test = "../PhobiaScan/data/Fer2013Dataset/test"

# # dir_train = "../data/images2/FER2013Train"
# # dir_test = "../data/images2/FER2013Test"
# # dir_valid = "../data/images2/FER2013Valid"

# datagen_train = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=10,
#     zoom_range=.1,
#     horizontal_flip=True,
#     rescale=1./255
# )

datagen_valid = ImageDataGenerator(
    rescale=1./255  # Sem augmentations para validação
)

# train_generator = datagen_train.flow_from_directory(
#     directory=dir_train,
#     target_size=(48, 48),
#     batch_size=32,
#     color_mode="grayscale",
#     class_mode="categorical"  # sem validation_split
# )

validation_generator = datagen_valid.flow_from_directory(
    directory=dir_test,  # usando diretório de validação dedicado
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical"
)

# # Para avaliação final (opcional):
# test_generator = datagen_valid.flow_from_directory(
#     directory=dir_test,
#     target_size=(48, 48),
#     batch_size=32,
#     color_mode="grayscale",
#     class_mode="categorical",
#     shuffle=False  # importante para avaliação
# )

# import matplotlib.pyplot as plt

# for images, _ in train_generator:
#     plt.imshow(images[0])
#     plt.show()
#     break

# input_shape = (48, 48, 1)
# num_classes = 5
# l2_regularization = 0.001
# patience = 100

# from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D
# from keras.models import Model
# from keras.regularizers import l2
# from keras import layers

# regularization = l2(l2_regularization)

# img_input = Input(input_shape)

# x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#             use_bias=False)(img_input)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)
# x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
#             use_bias=False)(x)
# x = BatchNormalization()(x)
# x = Activation('relu')(x)

# filter_list = [16, 32, 64, 128, 256]
# for n in filter_list:
#     residual = Conv2D(n, (1, 1), strides=(2, 2),
#                         padding='same', use_bias=False)(x)
#     residual = BatchNormalization()(residual)

#     x = SeparableConv2D(n, (3, 3), padding='same',
#                     depthwise_regularizer=regularization,
#                     pointwise_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)
#     x = Activation('relu')(x)
#     x = SeparableConv2D(n, (3, 3), padding='same',
#                     depthwise_regularizer=regularization,
#                     pointwise_regularizer=regularization,
#                         use_bias=False)(x)
#     x = BatchNormalization()(x)

#     x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
#     x = layers.add([x, residual])

# x = Conv2D(num_classes, (3, 3),
#             # kernel_regularizer=regularization,
#             padding='same')(x)
# x = GlobalAveragePooling2D()(x)
# output = Activation('softmax', name='predictions')(x)

# model = Model(img_input, output)

# model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# from keras.callbacks import ModelCheckpoint, EarlyStopping

# early_stop = EarlyStopping(
#     monitor='val_loss',
#     patience=patience,   
#     restore_best_weights=True
# )

# checkpoint = ModelCheckpoint(
#     "melhor_modelo.keras",
#     monitor='val_accuracy',
#     save_best_only=True,
#     verbose=1
# )

# hist = model.fit(
#     train_generator,
#     steps_per_epoch=len(train_generator),
#     epochs=1000,
#     verbose=1,
#     validation_data=validation_generator,
#     validation_steps=len(validation_generator),
#     callbacks=[early_stop, checkpoint]
# )


# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='upper right')
# plt.show()

# plt.plot(hist.history['accuracy'])
# plt.plot(hist.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Acurácia')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Val'], loc='lower right')
# plt.show()

# model.save("modelo_emocoes.keras")

from keras.models import load_model
model = load_model("modelo_emocoes.keras")

import matplotlib.pyplot as plt
# my_image = plt.imread("../data/FER2013Cleaned/test/fear/fer0000450.png")
my_image = plt.imread("../PhobiaScan/data/gab/Color/gab.153.jpg")
plt.imshow(my_image)

from skimage.transform import resize
my_image_resized = resize(my_image, (48,48,3))

img = plt.imshow(my_image_resized)

import numpy as np
from skimage.color import rgb2gray

my_image_gray = rgb2gray(my_image_resized) 
my_image_gray = np.expand_dims(my_image_gray, axis=-1)  
my_image_gray = np.expand_dims(my_image_gray, axis=0)

probabilities = model.predict(my_image_gray)

plt.imshow(np.squeeze(my_image_gray))
plt.show()

number_to_class = ['angry', 'disgust', 'fear', 'neutral', 'surprise']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[4]], "-- Probability:", probabilities[0,index[4]])
print("Second most likely class:", number_to_class[index[3]], "-- Probability:", probabilities[0,index[3]])
print("Third most likely class:", number_to_class[index[2]], "-- Probability:", probabilities[0,index[2]])
print("Fourth most likely class:", number_to_class[index[1]], "-- Probability:", probabilities[0,index[1]])
print("Fifth most likely class:", number_to_class[index[0]], "-- Probability:", probabilities[0,index[0]])

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# Gere predições no conjunto de validação
y_pred = model.predict(validation_generator)
y_pred = np.argmax(y_pred, axis=1)

# Matriz de confusão
print(confusion_matrix(validation_generator.classes, y_pred))

# Relatório por classe
print(classification_report(validation_generator.classes, y_pred,
                           target_names=number_to_class))