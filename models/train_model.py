import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight

from models.cnn import mini_Xception

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

random.seed(6743)
np.random.seed(6743)
tf.random.set_seed(6743)

print(tf.config.list_physical_devices('GPU'))

batch_size = 64
num_epochs = 10000
input_shape = (48, 48, 1)
verbose = 1
patience = 10
# min_lr = 1e-6
# factor = 0.2
# patienceReduce = int(patience/2)
patienceReduce = int(patience/4)

name = 'AffectnetGray'
# name = 'Fer2013AffectnetGray'
train_dir = 'data/' + name
# name = 'Fer2013'
# train_dir = 'data/' + name + '/train'
test_dir = 'data/' + name + '/test'

def trainAffectnet():
    # Aumento de Dataset
    data_generator_train = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=10,
        horizontal_flip=True,
        featurewise_center=False,
        featurewise_std_normalization=False,
        brightness_range=[0.3, 1.2],
        zoom_range=0.2,
        fill_mode='nearest',
        rescale=1./255,
        validation_split = 0.2
    )

    data_generator_test = ImageDataGenerator(
        rescale=1./255,
        validation_split = 0.2
    )

    # Disctribuição de 80/20 para trinamento e validação
    train_generator = data_generator_train.flow_from_directory(
        directory=train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset='training',
        shuffle=True
    )

    validation_generator = data_generator_test.flow_from_directory(
        directory=train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_generator, validation_generator

def trainFer2013():
    # Aumento de Dataset
    data_generator_train = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rotation_range=10,
        horizontal_flip=True,
        featurewise_center=False,
        featurewise_std_normalization=False,
        brightness_range=[0.3, 1.2],
        zoom_range=0.2,
        fill_mode='nearest',
        rescale=1./255
    )

    data_generator_test = ImageDataGenerator(
        rescale=1./255
    )

    # Disctribuição de 80/20 para trinamento e validação
    train_generator = data_generator_train.flow_from_directory(
        directory=train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True
    )

    validation_generator = data_generator_test.flow_from_directory(
        directory=test_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, validation_generator

def trainModel(l2, taxaDropout, factor, min_lr):
    if name == 'Fer2013':
        train_generator, validation_generator = trainFer2013()
    else:
        train_generator, validation_generator = trainAffectnet()

    num_classes = len(train_generator.class_indices)

    model = mini_Xception(num_classes, input_shape, l2, taxaDropout)

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    
    dir = 'models/checkpoint/' + name

    tamanho = 0

    while os.path.exists(f"{dir}/{name}_{tamanho}"):
        tamanho += 1

    newDir = f'{dir}/{name}_{tamanho}'
    os.makedirs(newDir,exist_ok=True)

    model_names = f"{newDir}/{name}" + ".{epoch:02d}-{val_accuracy:.2f}.keras"

    # Callbacks Parametros
    early_stop = EarlyStopping(
        'val_loss',
        patience=patience,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        model_names,
        'val_loss',
        verbose=verbose,
        save_best_only=True
    )

    reduce_lr = ReduceLROnPlateau(
        'val_loss',
        factor=factor,
        patience=patienceReduce,
        min_lr=min_lr,
        verbose=verbose
    )

    y_train = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weight_dict = dict(enumerate(class_weights))
    print(class_weight_dict)
    class_weight_dict[0] *= 2

    hist = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epochs,
        verbose=verbose,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # return hist, model, validation_generator, train_generator, name, min_lr, patience, batch_size, factor
    return hist, model, validation_generator, train_generator, name, patience, patienceReduce, batch_size