import keras
import tensorflow as tf
import keras_resnet.models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from keras import Model
from keras.layers import Lambda
import argparse
import tensorflow_addons as tfa
from keras import backend as K
from focal_loss import BinaryFocalLoss
from resnet1d import *

def recall_m(y_true, y_pred): # TPR
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # TP
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) # P
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # TP
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1))) # TP + FP
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_x", type=str,required=True, help='Train dataset X path')
    parser.add_argument("--train_y", type=str,required=True, help='Train dataset y path')
    parser.add_argument("--train_sex", type=str,required=False, help='Train dataset patient sex path')

    parser.add_argument("--val_x", type=str,required=True, help='Val dataset X path')
    parser.add_argument("--val_y", type=str,required=True, help='Val dataset y path')
    parser.add_argument("--val_sex", type=str,required=False, help='Val dataset patient sex path')

    parser.add_argument("--test_x", type=str,required=False, help='Test dataset X path')
    parser.add_argument("--test_y", type=str,required=False, help='Test dataset y path')
    parser.add_argument("--test_sex", type=str,required=False, help='Test dataset patient sex path')

    parser.add_argument("--output", type=str,required=True, help='output path')

    args=parser.parse_args()

    return args


def create_model(input_shape, top='flatten', add_gender=True):

    BottleneckLayer = {
        'flatten': Flatten()
    }[top]

    base = ResNet1D50(keras.layers.Input(input_shape),include_top=False)
    x = BottleneckLayer(base.output)
    #print(x)
    #x = Flatten()(x)
    #x = GlobalAveragePooling1D()(x)
    if add_gender:
        gender = keras.layers.Input((2,))
        x = tf.keras.layers.Concatenate(axis=1)([x,gender])
    x = BatchNormalization()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)


    x = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=[base.inputs, gender], outputs=x)
    return model

def main():
    
    inputs = parse_args()

    X_train = np.load(inputs.train_x)
    y_train = np.load(inputs.train_y)

    if inputs.train_sex is not None:
        sex_train = np.load(inputs.train_sex)
        add_gender = True
    else:
        add_gender = False


    X_val = np.load(inputs.val_x)
    y_val = np.load(inputs.val_y)


    if inputs.val_sex is not None:
        sex_val = np.load(inputs.val_sex)


    model = create_model(input_shape=(500,12),add_gender=add_gender)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=BinaryFocalLoss(gamma=2),metrics=[
    tf.keras.metrics.AUC(
        num_thresholds=200,
        curve='PR',
        summation_method='interpolation',
        name=None,
        dtype=None,
        thresholds=None,
        multi_label=False,
        num_labels=None,
        label_weights=None,
        from_logits=False
        ),'binary_accuracy','categorical_accuracy',recall_m, precision_m,f1_m])

    mcp_save = ModelCheckpoint(filepath=filepath, save_best_only=True, monitor='val_loss', mode='min')

    lrplateau = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=2,
        verbose=0,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=1e-12)

    if add_gender:
        model.fit([X_train,sex_train], y_train, verbose=1, epochs=50, callbacks=[lrplateau,mcp_save], validation_data=([X_val,sex_val],y_val), batch_size=96)

    else:
        model.fit(X_train, y_train, verbose=1, epochs=50, callbacks=[lrplateau,mcp_save], validation_data=(X_val,y_val), batch_size=96)

    if inputs.test_sex is not None:
        X_test = np.load(inputs.test_x)
        y_test = np.load(inputs.test_y)
        sex_test = np.load(inputs.test_sex)

        out = model.evaluate([X_test,sex_test], y_test, batch_size=128)  
        print('test loss: {}'.format(out[0]))
        print('AUPR: {}'.format(out[1]))
        print('binary_accuracy: {}'.format(out[2]))
        print('categorical_accuracy: {}'.format(out[3]))
        print('recall: {}'.format(out[4]))
        print('precision: {}'.format(out[5]))
        print('F1: {}'.format(out[6]))

if __name__ == '__main__':
    main()