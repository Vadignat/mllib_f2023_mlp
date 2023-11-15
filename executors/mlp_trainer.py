import os

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from datasets.KMNIST import KMNIST
from models.MLP import MLP
from datasets.utils.prepare_transforms import prepare_transforms
from utils.metrics import accuracy
from utils.visualization import confusion_matrix
from logs.Logger import Logger


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # TODO: настройте логирование с помощью класса Logger
        #  (пример: https://github.com/KamilyaKharisova/mllib_f2023/blob/master/logginig_example.py)

        # TODO: залогируйте используемые гиперпараметры в neptune.ai через метод log_hyperparameters

        self.__prepare_data(self.cfg.dataset_cfg)
        self.__prepare_model(self.cfg.model_cfg)

    def __prepare_data(self, dataset_cfg):
        """ Подготовка обучающих и тестовых данных """
        self.train_dataset = KMNIST(dataset_cfg, 'train',
                                    transforms=prepare_transforms(dataset_cfg.transforms['train']))
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True)

        self.test_dataset = KMNIST(dataset_cfg, 'test', transforms=prepare_transforms(dataset_cfg.transforms['test']))
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.cfg.batch_size, shuffle=False)

    def __prepare_model(self, model_cfg):
        """ Подготовка нейронной сети"""
        self.model = MLP(model_cfg)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: инициализируйте оптимайзер через getattr(torch.optim, self.cfg.optimizer_name)
        self.optimizer = None

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        raise NotImplementedError

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        raise NotImplementedError

    def make_step(self, batch, update_model=True):
        """
            Этот метод выполняет один шаг обучения, включая forward pass, вычисление целевой функции,
            backward pass и обновление весов модели (если update_model=True).

            :param batch: dict of data with keys ["image", "label"]
            :param update_model: bool - если True, необходимо сделать backward pass и обновить веса модели
            :return: значение функции потерь, выход модели
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        raise NotImplementedError

    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            TODO: реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции и accuracy на batch
        """
        self.model.train()
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        """
            Метод используется для проверки производительности модели на обучающих/тестовых данных. Сначала модель
            переводится в режим оценки (model.eval()), затем данные последовательно подаются на вход модели, по
            полученным выходам вычисляются метрики производительности, такие как значение целевой функции, accuracy

            TODO: реализуйте функцию оценки с использованием метода self.make_step(batch, update_model=False),
                залогируйте значения целевой функции и accuracy, постройте confusion_matrix
        """
        self.model.eval()
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # TODO: реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        raise NotImplementedError

    def overfitting_on_batch(self, max_step=100):
        """
            Оверфиттинг на одном батче. Эта функция может быть полезна для отладки и оценки способности вашей
            модели обучаться и обновлять свои веса в ответ на полученные данные.
        """
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 10 == 0:
                acc = accuracy(output, batch['label'])
                print('[{:d}]: loss - {:.4f}, {:.4f}'.format(step, loss, acc))


if __name__ == '__main__':
    from configs.train_cfg import cfg

    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
