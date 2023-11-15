import matplotlib.pyplot as plt


def confusion_matrix(*args, **kwargs):
    """
        Построение и визуализация confusion matrix
            confusion_matrix - матрица NxN, где N - кол-во классов в наборе данных
            confusion_matrix[i, j] - кол-во элементов класса "i", которые классифицируются как класс "j"

        :return plt.gcf() - matplotlib figure
        TODO: реализуйте построение и визуализацию confusion_matrix, подпишите оси на полученной визуализации, добавьте значение confusion_matrix[i, j] в соотвествующие ячейки на изображении
    """

    raise NotImplementedError

    plt.cla(), plt.clf()
    plt.imshow(conf_matrix)
    return plt.gcf()
