import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize
from skimage.color import rgb2gray

# docker run --gpus all -it -v "C:\Users\gmara\Documents\Sourcetree\PhobiaScan:/tf/PhobiaScan" -w /tf/PhobiaScan tf-gpu-custom python -m src.fer2013


# docker run --gpus all -it -v "C:\Users\gmara\.nv:/root/.nv" -v "C:\Users\gmara\Documents\Sourcetree\PhobiaScan:/tf/PhobiaScan" -w /tf/PhobiaScan tf-gpu-custom python -m src.fer2013

from models.cnn import fine_tuning_Mini_Xception, mini_Xception

# Diretórios
dir_train = "../PhobiaScan/data/Fer2013/train"
dir_test = "../PhobiaScan/data/Fer2013/test"

# Data augmentation para treino e validação
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

# fine_tuning_Mini_Xception(train_generator, validation_generator)

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
patience = 50  
verbose = 1

model = mini_Xception(num_classes, input_shape)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_filename = f"models/chechpoint_fer2013_{timestamp}.keras"

checkpoint = ModelCheckpoint(
    checkpoint_filename,
    monitor='val_loss',
    save_best_only=True,
    verbose=verbose
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=(patience/4),
    min_lr=0.00001,
    verbose=verbose
)

y_train = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weight_dict = dict(enumerate(class_weights))

hist = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    verbose=verbose,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    class_weight=class_weight_dict,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

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

final_model_filename = f"models/fer2013_{timestamp}.keras"
model.save(final_model_filename)
print(f"Modelo salvo como {final_model_filename}")

model = load_model(checkpoint_filename)

from skimage.io import imread

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
plt.imshow(my_image_gray.squeeze(), cmap='gray')
plt.show()

number_to_class = list(train_generator.class_indices.keys())

index = np.argsort(probabilities[0, :])[::-1]
for i in range(min(5, len(number_to_class))):
    print(f"{i+1}ª classe mais provável: {number_to_class[index[i]]} -- Probabilidade: {probabilities[0, index[i]]:.3f}")

y_test_pred = model.predict(test_generator)
y_test_pred = np.argmax(y_test_pred, axis=1)
y_test_true = test_generator.classes

number_to_class = list(test_generator.class_indices.keys())

print("Matriz de confusão (Teste):")
print(confusion_matrix(y_test_true, y_test_pred))
print("Relatório de classificação (Teste):")
print(classification_report(y_test_true, y_test_pred, target_names=number_to_class))

import seaborn as sns
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
plt.title('Matriz de Confusão')
plt.savefig('confusion_matrix.png')
plt.show()