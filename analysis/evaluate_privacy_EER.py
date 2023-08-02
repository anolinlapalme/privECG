
import keras
import tensorflow as tf
import keras_resnet.models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from keras import Model
from keras.layers import Lambda
import argparse
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_x", type=str, required=True, help='siamese train dataset X path')
    parser.add_argument("--test_y", type=str, required=True, help='siamese train dataset y path')
    parser.add_argument("--model_path", type=str, required=True, help='siamese val dataset X path')

    args=parser.parse_args()

    return args


def main():
    
    inputs = parse_args()
    model =  keras.models.load_model(inputs.model_path)

    X_test = np.load(inputs.test_x)
    x_test_1,x_test_2 = X_test
    y_pred = model.predict([x_test_1,x_test_2])
    y_test = np.load(inputs.test_y)
    y_pred = np.squeeze(y_pred)

    y_pred_sigmoid = list()
    for i in y_pred:
        if i < 0.5:
            y_pred_sigmoid.append(0)
        else:
            y_pred_sigmoid.append(1)

    accuracy = accuracy_score(y_test, y_pred_sigmoid)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_sigmoid, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    print('Test accuracy at reidentification {}'.format(accuracy))
    print('EER {}'.format(eer))

if __name__ == '__main__':
    main()

