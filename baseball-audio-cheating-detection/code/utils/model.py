import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


# Custom CNN subclass and model architecture

class EarlyStoppingWithThreshold(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss',patience=0,threshold=1):
        super(EarlyStoppingWithThreshold, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.best_weights = None
        self.best_metric = -np.inf
        self.wait = 0
        self.threshold = threshold


    def on_epoch_end(self, epoch, logs=None):
        current_metric = logs.get(self.monitor)

        # don't do anything if the metric hasn't breached the threshold
        if np.less(current_metric, self.threshold):
            self.wait = 0
            return

        if np.greater(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        print("Stopped early")


class PitchModel(tf.keras.Model):
    def __init__(self, image_shape):

        super(PitchModel, self).__init__()

        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu',input_shape=image_shape,name='conv_layer_1')
        self.mp1 = layers.MaxPooling2D(pool_size=(2, 2),name='max_pooling_1')
        self.drop1 = layers.Dropout(0.1,name='dropout_1')
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu',name='conv_layer_2')
        self.mp2 = layers.MaxPooling2D(pool_size=(2, 2),name='max_pooling_2')
        self.drop2 = layers.Dropout(0.1,name='dropout_2')
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu',name='conv_layer_3')
        self.mp3 = layers.GlobalMaxPooling2D(name='global_max_pooling')
        self.flat = layers.Flatten(name='flatten')
        self.dense1 = layers.Dense(124, activation='relu',name='image_dense_layer')
        self.dense2 = layers.Dense(4, activation='relu',name='metadata_dense_layer')
        self.combined = tf.keras.layers.Concatenate(name='concatenate_layers')
        self.dense3 = layers.Dense(128, activation='relu',name='final_dense_layer')
        self.drop3 = layers.Dropout(0.1,name='dropout_3')
        self.out = layers.Dense(1, activation='sigmoid',name='output_layer')



    def call(self, inputs):

        # input layers
        image_input = inputs[0]
        meta_data_input = inputs[1]

        # convolutional layer 1
        x1 = self.conv1(image_input)
        x1 = self.mp1(x1)
        x1 = self.drop1(x1)

        # convolutional layer 2
        x1 = self.conv2(x1)
        x1 = self.mp2(x1)
        x1 = self.drop2(x1)

        # convolutional layer 3
        x1 = self.conv3(x1)
        x1 = self.mp3(x1)

        # dense layer for convolutional layers
        x1 = self.flat(x1)
        x1= self.dense1(x1)

        # dense layer for the meta data
        x2 = self.dense2(meta_data_input)
        x = self.combined([x1,x2])

        # dense layer on the combined data
        x = self.dense3(x)
        x = self.drop3(x)

        # output layer
        x = self.out(x)

        return x