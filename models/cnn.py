from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import layers
from tensorflow.keras.models import load_model

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


def mini_Xception(num_classes, input_shape, l2_regularization = 0.01):
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
        x = Dropout(0.3)(x)
        x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), 
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    model.summary()

    return model

def fine_tuning_Mini_Xception(train_generator, validation_generator):
    base_model = load_model('models/mini_Xception/fer2013_mini_XCEPTION.119-0.65.hdf5', compile=False)
    base_model.trainable = False

    # Obter as classes do gerador
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    # Identificar os índices das classes prioritárias
    priority_classes = ['disgust', 'fear', 'neutral', 'surprise']
    priority_indices = [class_indices[class_name] for class_name in priority_classes if class_name in class_indices]

    print(f"Classes prioritárias e seus índices: {[(class_names[idx], idx) for idx in priority_indices]}")

    # Congelar todas as camadas do modelo base
    for layer in base_model.layers[:-2]:  # Manter as últimas camadas treináveis
        layer.trainable = False

    # Obter a saída da penúltima camada
    x = base_model.layers[-2].output

    # Substituir a camada de classificação final
    predictions = Dense(len(class_indices), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Definir pesos para as classes (dar mais peso às classes prioritárias)
    class_weights = {}
    for i in range(len(class_indices)):
        if i in priority_indices:
            class_weights[i] = 3.0  # Peso maior para classes prioritárias
        else:
            class_weights[i] = 1.0  # Peso normal para outras classes
    
    # Fase 1: Treinar apenas a camada final
    history_top = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        class_weight=class_weights
    )

    # Fase 2: Fine-tuning - descongelar algumas camadas superiores
    for layer in model.layers[-10:]:  # Descongelar as últimas 10 camadas
        layer.trainable = True

    # Recompilar com taxa de aprendizado menor
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Taxa menor para fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Continuar treinamento com fine-tuning
    history_ft = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        class_weight=class_weights
    )

    # Salvar o modelo final
    model.save('mini_xception_finetuned.h5')

    # Avaliar o modelo no conjunto de validação
    results = model.evaluate(validation_generator)
    print(f"Perda: {results[0]}, Precisão: {results[1]}")

    # Fazer previsões no conjunto de validação
    validation_generator.reset()
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)

    # Obter rótulos reais
    y_true = validation_generator.classes

    # Calcular métricas por classe
    from sklearn.metrics import classification_report, confusion_matrix

    print("\nRelatório de classificação:")
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=[class_names[i] for i in range(len(class_names))],
        digits=4
    )
    print(report)

    # Métricas específicas para classes prioritárias
    print("\nDesempenho nas classes prioritárias:")
    for idx in priority_indices:
        class_name = class_names[idx]
        mask_true = (y_true == idx)
        mask_pred = (y_pred == idx)
        
        true_positives = np.sum(mask_true & mask_pred)
        false_positives = np.sum(~mask_true & mask_pred)
        false_negatives = np.sum(mask_true & ~mask_pred)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name}: Precisão={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Plotar matriz de confusão
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    # Combinar históricos de treinamento
    def combine_history(h1, h2):
        history = {}
        for k in h1.history.keys():
            history[k] = h1.history[k] + h2.history[k]
        return history

    history = combine_history(history_top, history_ft)

    # Plotar gráficos de precisão e perda
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Precisão do Modelo')
    plt.ylabel('Precisão')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper right')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
