import tensorflow as tf
from deep_learning_models import MyCustomModel
from my_utils import display_some_examples
import numpy as np
import tensorflow.keras.utils

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False:
        display_some_examples(x_train, y_train)

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    model = MyCustomModel()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")
    # Model training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    # Evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)