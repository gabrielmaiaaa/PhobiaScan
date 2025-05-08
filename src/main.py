from tensorflow.keras.preprocessing.image import ImageDataGenerator

dir_train = "../PhobiaScan/data/Fer2013/train"
dir_test = "../PhobiaScan/data/Fer2013/test"

# # dir_train = "../data/Fer2013Plus/FER2013Train"
# # dir_test = "../data/Fer2013Plus/FER2013Test"
# # dir_valid = "../data/Fer2013Plus/FER2013Valid"

datagen_train = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,  
    zoom_range=0.15,   
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  
    rescale=1./255,
    validation_split=0.2
)


datagen_valid = ImageDataGenerator(
    rescale=1./255,       
)

train_generator = datagen_train.flow_from_directory(
    directory=dir_train,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)
validation_generator = datagen_train.flow_from_directory(
    directory=dir_train,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

import matplotlib.pyplot as plt

for images, _ in train_generator:
    plt.imshow(images[0])
    plt.show()
    break

input_shape = (48, 48, 1)
num_classes = len(train_generator.class_indices)
l2_regularization = 0.01
patience = 100
verbose = 1

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
from keras import layers
from keras.layers import Dropout

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
x = Dropout(0.25)(x)

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
    x = Dropout(0.25)(x)
    x = SeparableConv2D(n, (3, 3), padding='same',
                    depthwise_regularizer=regularization,
                    pointwise_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

x = Conv2D(num_classes, (3, 3),
            # kernel_regularizer=regularization,
            padding='same')(x)
x = GlobalAveragePooling2D()(x)
output = Activation('softmax', name='predictions')(x)

model = Model(img_input, output)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,   
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "melhor_modelo.keras",
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

y_train = np.array(train_generator.classes)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))

hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    verbose=verbose,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint, reduce_lr]
)


import matplotlib.pyplot as plt
from datetime import datetime

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

# Crie o timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Monte o nome do arquivo com o timestamp
model_filename = f"modelo_emocoes_{timestamp}.keras"

# Salve o modelo
model.save(model_filename)


from keras.models import load_model
model = load_model(model_filename)

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

number_to_class = list(train_generator.class_indices.keys())

index = np.argsort(probabilities[0, :])[::-1]
for i in range(min(5, len(number_to_class))):
    print(f"{i+1}ª classe mais provável: {number_to_class[index[i]]} -- Probabilidade: {probabilities[0, index[i]]:.3f}")

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