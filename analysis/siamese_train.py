import keras
import tensorflow as tf
import keras_resnet.models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from keras import Model
from keras.layers import Lambda
import argparse
from tensorflow.keras.layers import Flatten
from resnet1d import *
from tensorflow.python.keras import backend as K
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_x", type=str, required=True, help='siamese train dataset X path')
    parser.add_argument("--train_y", type=str, required=True, help='siamese train dataset y path')
    parser.add_argument("--val_x", type=str, required=True, help='siamese val dataset X path')
    parser.add_argument("--val_y", type=str, required=True, help='siamese val dataset y path')
    parser.add_argument("--test_x", type=str, required=False, help='siamese val dataset X path')
    parser.add_argument("--test_y", type=str, required=False, help='siamese val dataset y path')
    parser.add_argument("--output", type=str, required=True, help='output path')

    args=parser.parse_args()

    return args


def create_model(input_shape, top='flatten'):

    BottleneckLayer = {
        'flatten': Flatten()
    }[top]

    input_1 = keras.layers.Input(input_shape)
    input_2 = keras.layers.Input(input_shape)

    base = ResNet1D50(keras.layers.Input(input_shape),include_top=False)
    x_1 = BottleneckLayer(input_1)
    x_2 = BottleneckLayer(input_2)

    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([x_1, x_2])

    outputs = Dense(1, activation="sigmoid")(L1_distance)
    
    model = Model(inputs=[input_1, input_2], outputs=outputs)
    return model

def main():

    inputs = parse_args()

    os.makedirs(inputs.output,exist_ok=True)

    model = create_model(input_shape=(500,12))

    X_train = np.load(inputs.train_x)
    y_train = np.load(inputs.train_y)
    x_train_1,x_train_2 = X_train

    X_val = np.load(inputs.val_x)
    y_val = np.load(inputs.val_y)
    x_val_1,x_val_2 = X_val

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    filepath = os.path.join(inputs.output,'siamese.epoch{epoch:02d}-loss{val_loss:.2f}.hdf5')
    mcp_save = ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='val_loss', mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=6, verbose=1)
    lrplateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=4,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=1e-14,
    )
    model.fit([x_train_1,x_train_2], y_train, verbose=1, epochs=250, callbacks=[early_stop,lrplateau,mcp_save], validation_data=([x_val_1,x_val_2],y_val), batch_size=64)

    if inputs.test_x is not None:
        X_test = np.load(inputs.test_x)
        y_test = np.load(inputs.val_y)
        x_test_1,x_test_2 = X_test

        out = model.evaluate([x_test_1,x_test_2], y_test, batch_size=128)  
        print('test loss: {} & test a accuracy: {}'.format(out[0],out[1]))

if __name__ == '__main__':
    main()

