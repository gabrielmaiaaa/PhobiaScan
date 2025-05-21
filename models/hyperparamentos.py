import os
import tensorflow as tf

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight

from models.cnn import mini_Xception

import keras_tuner as kt

import random
import numpy as np
import tensorflow as tf

random.seed(6743)
np.random.seed(6743)
tf.random.set_seed(6743)

def build_model(hp):
    l2_reg = hp.Float('l2', min_value=0.0001, max_value=0.05, sampling='log')
    dropout = hp.Float('dropout', min_value=0.2, max_value=0.7, step=0.1)
    learning_rate = hp.Float('learning_rate', min_value=1e-7, max_value=1e-2, sampling='log')
    num_classes = 4 
    input_shape = (48, 48, 1)
    model = mini_Xception(num_classes, input_shape, l2_reg, dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model

def hyperparametro():
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=120,
        directory='hiperparametros',
        project_name='Hyperband'
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_batch_size = best_hps.Choice('batch_size', [16, 32, 64])
    best_factor = best_hps.Float('factor', min_value=0.1, max_value=0.5, step=0.1)
    best_min_lr = best_hps.Float('min_lr', min_value=1e-6, max_value=1e-4, sampling='log')

    if name == 'Fer2013':
        train_generator, validation_generator = trainFer2013()
    else:
        train_generator, validation_generator = trainAffectnet()


    y_train = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))


    early_stop = EarlyStopping(
        'val_loss',
        patience=patience,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        'val_loss',
        factor=best_factor,
        patience=int(patience/4),
        min_lr=best_min_lr
    )

    # tuner = kt.Hyperband(
    #     build_model,
    #     objective='val_accuracy',
    #     max_epochs=120,
    #     directory='hiperparametros',
    #     project_name='Hyperband'
    # )
    
    # tuner = kt.RandomSearch(
    #     build_model,
    #     objective='val_accuracy',
    #     max_trials=10,
    #     executions_per_trial=2,
    #     directory='hiperparametros',
    #     project_name='RandomSearch'
    # )

    tuner.search(
        train_generator,
        epochs=120,
        validation_data=validation_generator,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Melhores hiperparâmetros:")
    print(f"L2={best_hps.get('l2')}")
    print(f"Dropout={best_hps.get('dropout')}")
    print(f"Learning rate={best_hps.get('learning_rate')}")
    print(f"Batch size={best_batch_size}")
    print(f"ReduceLROnPlateau factor={best_factor}")
    print(f"ReduceLROnPlateau min_lr={best_min_lr}")

# Best val_accuracy So Far: 0.601082444190979
# Total elapsed time: 08h 01m 46s
# Melhores hiperparâmetros: L2=0.002789250672351807,
# Dropout=0.2