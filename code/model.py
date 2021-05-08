import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, concatenate, Input, Activation, Dropout, Dense
from tensorflow.keras import Model

def build_model():
    # input layer
    inputs = Input(shape = (280, 480, 1)) # channel dimension last
    default_size = 32

    conv1 = Conv2D(default_size, kernel_size = 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Activation("relu")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1s = Conv2D(default_size, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1s = Activation("relu")(conv1s)
    pool1 = BatchNormalization()(conv1s)
    pool1 = MaxPooling2D(pool_size=(2, 2))(pool1)#140

    conv2 = Conv2D(default_size * 2, kernel_size = 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Activation("relu")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2s = Conv2D(default_size * 2, kernel_size = 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2s = Activation("relu")(conv2s)
    pool2 = BatchNormalization()(conv2s)
    pool2 = MaxPooling2D(pool_size=(2, 2))(pool2)#70

    conv3 = Conv2D(default_size * 4, kernel_size = 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Activation("relu")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3s = Conv2D(default_size * 4, kernel_size = 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3s = Activation("relu")(conv3s)
    pool3 = BatchNormalization()(conv3s)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3s)#35

    conv4 = Conv2D(default_size * 5, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Activation("relu")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4s = Conv2D(default_size * 5, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4s = Activation("relu")(conv4s)
    conv4s = BatchNormalization()(conv4s)
    drop4 = Dropout(0.5)(conv4s)

    up5 = concatenate([Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(drop4),conv3s], axis=3)
    conv5 = Conv2D(128, kernel_size = 3, padding='same', kernel_initializer='he_normal')(up5)
    conv5 = Activation("relu")(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, kernel_size = 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(conv5),conv2s], axis=3)
    conv6 = Conv2D(64, kernel_size = 3, padding='same', kernel_initializer='he_normal')(up6)
    conv6 = Activation("relu")(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, kernel_size = 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(conv6),conv1s], axis=3)
    conv7 = Conv2D(32, kernel_size = 3, padding='same', kernel_initializer='he_normal')(up7)
    conv7 = Activation("relu")(conv7)
    conv7 = BatchNormalization()(conv7)
    conv8 = Conv2D(32, kernel_size = 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv8 = Activation("relu")(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, kernel_size = 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)


    pool5 = MaxPooling2D(pool_size=(2, 2))(conv8)
    pool5 = tf.keras.layers.Flatten()(pool5)

    hidden1 = Dense(128, activation = 'relu')(pool5)
    hidden2 = Dense(64,  activation = 'relu')(hidden1)

    outputs = Dense(5, activation = 'softmax')(hidden2)
    model = Model(inputs=inputs, outputs=outputs)

    print(model.summary())
    return model
