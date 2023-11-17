import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(predictions, labels):
    """
        Построение и визуализация confusion matrix
            confusion_matrix - матрица NxN, где N - кол-во классов в наборе данных
            confusion_matrix[i, j] - кол-во элементов класса "i", которые классифицируются как класс "j"

        :return plt.gcf() - matplotlib figure
        TODO: реализуйте построение и визуализацию confusion_matrix, подпишите оси на полученной визуализации, добавьте значение confusion_matrix[i, j] в соотвествующие ячейки на изображении
    """

    num_classes = max(max(predictions), max(labels)) + 1
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int)

    for i in range(len(predictions)):
        conf_matrix[labels[i], predictions[i]] += 1

    plt.cla(), plt.clf()
    plt.imshow(conf_matrix)
    return plt.gcf()
