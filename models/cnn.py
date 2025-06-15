from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, SeparableConv2D, Input, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras import layers

def mini_Xception(num_classes, input_shape, l2_regularization, drop):
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
        x = Dropout(drop)(x)
        x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3), 
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    model.summary()

    return model