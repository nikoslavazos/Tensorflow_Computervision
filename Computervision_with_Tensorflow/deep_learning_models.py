import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from tensorflow.keras import Model
import numpy as np



class MyCustomModel(tf.keras.Model):

        def __init__(self):
                super().__init__()

                self.conv1 = Conv2D(32, (3, 3), activation="relu")
                self.conv2 = Conv2D(64, (3, 3), activation="relu")
                self.maxpool1 = MaxPool2D()
                self.batchnorm1 = BatchNormalization()

                self.conv3 = Conv2D(128, (3, 3), activation="relu")
                self.maxpool2 = MaxPool2D()
                self.batchnorm2 = BatchNormalization()

                self.globalavgpool1 = GlobalAvgPool2D()
                self.dense1 = Dense(64, activation="relu")
                self.dense2 = Dense(10, activation="softmax")


        def call(self, myinput):

                x = self.conv1(myinput)
                x = self.conv2(x)
                x = self.maxpool1(x)
                x = self.batchnorm1(x)
                x = self.conv3(x)
                x = self.maxpool2(x)
                x = self.batchnorm2(x)
                x = self.globalavgpool1(x)
                x = self.dense1(x)
                x = self.dense2(x)

                return x


def streetsigns_model(nbr_classes):

        my_input = Input(shape=(60, 60, 3))

        x = Conv2D(32, (3, 3), activation="relu")(my_input)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, (3, 3), activation="relu")(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Conv2D(128, (3, 3), activation="relu")(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        # x = GlobalAvgPool2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(nbr_classes, activation="softmax")(x)

        return Model(inputs=my_input, outputs=x)


if __name__ == "__main__":

        model = streetsigns_model(10)
        model.summary()

