from functools import partial

import keras.backend as K
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import to_categorical


def normalize_y_pred(y_pred):
    return K.one_hot(K.argmax(y_pred), y_pred.shape[-1])


def class_true_positive(class_label, y_true, y_pred):
    y_pred = normalize_y_pred(y_pred)
    return K.cast(K.equal(y_true[:, class_label] + y_pred[:, class_label], 2),
                  K.floatx())


def class_accuracy(class_label, y_true, y_pred):
    y_pred = normalize_y_pred(y_pred)
    return K.cast(K.equal(y_true[:, class_label], y_pred[:, class_label]),
                  K.floatx())


def class_precision(class_label, y_true, y_pred):
    y_pred = normalize_y_pred(y_pred)
    return K.sum(class_true_positive(class_label, y_true, y_pred)) / (K.sum(y_pred[:, class_label]) + K.epsilon())


def class_recall(class_label, y_true, y_pred):
    return K.sum(class_true_positive(class_label, y_true, y_pred)) / (K.sum(y_true[:, class_label]) + K.epsilon())


def class_f_measure(class_label, y_true, y_pred):
    precision = class_precision(class_label, y_true, y_pred)
    recall = class_recall(class_label, y_true, y_pred)
    return (2 * precision * recall) / (precision + recall + K.epsilon())


def true_positive(y_true, y_pred):
    y_pred = normalize_y_pred(y_pred)
    return K.cast(K.equal(y_true + y_pred, 2),
                  K.floatx())


def micro_precision(y_true, y_pred):
    y_pred = normalize_y_pred(y_pred)
    return K.sum(true_positive(y_true, y_pred)) / (K.sum(y_pred) + K.epsilon())


def micro_recall(y_true, y_pred):
    return K.sum(true_positive(y_true, y_pred)) / (K.sum(y_true) + K.epsilon())


def micro_f_measure(y_true, y_pred):
    precision = micro_precision(y_true, y_pred)
    recall = micro_recall(y_true, y_pred)
    return (2 * precision * recall) / (precision + recall + K.epsilon())


def average_accuracy(y_true, y_pred):
    class_count = y_pred.shape[-1]
    class_acc_list = [class_accuracy(i, y_true, y_pred) for i in range(class_count)]
    class_acc_matrix = K.concatenate(class_acc_list, axis=0)
    return K.mean(class_acc_matrix, axis=0)


def macro_precision(y_true, y_pred):
    class_count = y_pred.shape[-1]
    return K.sum([class_precision(i, y_true, y_pred) for i in range(class_count)]) \
           / K.cast(class_count, K.floatx())


def macro_recall(y_true, y_pred):
    class_count = y_pred.shape[-1]
    return K.sum([class_recall(i, y_true, y_pred) for i in range(class_count)]) \
           / K.cast(class_count, K.floatx())


def macro_f_measure(y_true, y_pred):
    precision = macro_precision(y_true, y_pred)
    recall = macro_recall(y_true, y_pred)
    return (2 * precision * recall) / (precision + recall + K.epsilon())


def generate_metrics():
    metrics = ["accuracy"]

    # classごとのmetrics
    func_list = [class_accuracy, class_precision, class_recall, class_f_measure]
    name_list = ["acc", "precision", "recall", "f_measure"]
    for i in range(10):
        for func, name in zip(func_list, name_list):
            func = partial(func, i)
            func.__name__ = "{}-{}".format(name, i)
            metrics.append(func)

    # 全体のmetrics
    metrics.append(average_accuracy)
    metrics.append(macro_precision)
    metrics.append(macro_recall)
    metrics.append(macro_f_measure)

    return metrics


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255
    x_test = x_test.reshape(10000, 784) / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    inputs = Input(shape=(784,))
    x = Dense(32, activation="relu")(inputs)
    x = Dense(10, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=x)

    model.compile(optimizer="SGD",
                  loss="categorical_crossentropy",
                  metrics=generate_metrics())

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=400,
              epochs=50,
              callbacks=[TensorBoard()])


if __name__ == "__main__":
    main()
