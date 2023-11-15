### Часть 1:
- Класс датасета KMNIST (функции `__len__`, `__getitem__`, `show_statistic`)
- Класс нейронной сети MLP (функции `_init_layers`, `_init_weights`, `forward`) 
- Подсчет количества обучаемых параметров сети
- Сохранение и загрузка весов модели – через torch.save() и torch.load()
- Инференс нейронной сети 

### Часть 2:
- Класс обучения нейронной сети Trainer (функции `make_step`, `train_epoch`, `evaluate`, `fit`)
- Логирование процесса и результатов обучения через Neptune.ai (logs/Logger.py)
- Функция подсчета метрики accuracy (utils/metrics.py) и построения confusion matrix (utils/visualization.py) без использования стороних библиотек (numpy, torch - можно)


## Многослойный персептрон:

### Набор данных KMNIST:
1) Реализация класс датасета KMNIST (функции `len`, `getitem`, `show_statistic`)
2) Выводить количества элементов в наборе данных, количество элементов в каждом классе
3) Посчитать среднее (mean) и стандартное отклонение (std) по всем нормализованным изображениям __обучающего__ набора данных, полученные значения использовать при применении transforms.Normalize() (и для train и для test set)

### MLP:
1) Реализация класса многослойного персептрона MLP (функции `_init_layers`, `_init_weights`, `forward`, `_prepare_model`)
2) Описание модели должно быть вынесено в конфиг (основной код не должен меняться в зависимости от эксперимента)
3) Все вычисления в методе forward производить для батча
4) Выводить количество обучаемых параметров сети (например, после инициализации модели)

### Обучение моделей:

1) Реализация класса обучения Trainer (функции `make_step`, `train_epoch`, `evaluate`, `fit`, `save_model`, `load_model`)
2) Логирование процесса и результатов обучения реализовать через neptune.ai (класс logs/Logger.py)
3) Реализация функций подсчета метрики accuracy (utils/metrics.py) без использования сторонних библиотек (numpy, torch - можно)
4) Построения confusion matrix (utils/visualization.py) без использования сторонних библиотек (numpy, torch - можно), визуализация confusion matrix через matplotlib и логирование в neptune.ai (через метод save_plot в классе Logger)
5) Во время обучения логировать значение целевой функции, accuracy на обучающей выборке на каждом шаге
6) После окончания каждой эпохи обучения посчитать значения целевой функции, accuracy на тестовых данных, полученные значения логировать в neptune.ai
7) Построить confusion matrix на тестовых данных в начале и в конце обучения, залогировать матрицы в neptune.ai в виде изображений (можно на каждой эпохе в evaluate)
8) Сохранить модель с лучшим значением accuracy на тестовой выборке (класс Trainer метод save_model)

## Этапы:
1) Добиться оверфиттинга на одном батче (метод overfitting_on_batch в классе Trainer)
2) Обучить модель с одним скрытым слоем размера 200 с функцией активации ReLU (базовая модель), подобрать learning rate, batch size
3) Добиться точности > 89% для базовой модели на тестовых данных (достаточно около 10 эпох)
4) Добавить аугментацию на обучающей выборке при обучении базовой модели и посмотреть, как изменится результат (Pad, RandomCrop и т.д.)
5) Дополнительно обучить минимум 5 моделей с различными наборами гиперпараметров: глубина нейронной сети (количество скрытых слоев), количество нейронов в скрытых слоях, функции активации, различные методы инициализации параметров сети и различные оптимайзеры (все параметры должны настраиваться в конфигах)
6) Сравнить базовую модель и модели, обученные в пункте 5 (можно добавить README.md к репозиторию с описанием наблюдений)

Все модели обучать до насыщения accuracy на тестовой выборке (обычно 10-20 эпох), каждый эксперимент и его параметры должны быть залогированы в neptune.ai (с понятным название эксперимента)

Обучение можно проводить локально на компьютере, либо воспользоваться [kaggle](https://www.kaggle.com/)/[colab](https://colab.research.google.com/).

### Dead line - 18 ноября



### Полезные ссылки:
- https://pytorch.org/docs/stable/nn.init.html
- https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
- https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
- https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- https://docs-python.ru/tutorial/vstroennye-funktsii-interpretatora-python/funktsija-getattr/
- https://pytorch.org/tutorials/beginner/saving_loading_models.html
- https://pytorch.org/vision/0.11/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
- https://pytorch.org/docs/stable/optim.html
