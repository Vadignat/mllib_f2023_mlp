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
        self.neptune_logger = Logger(cfg.env_path, cfg.project_name)
        self.neptune_logger.log_hyperparameters(params={
        'learning_rate': cfg.lr,
        'batch_size': cfg.batch_size,
        'optimizer': cfg.optimizer_name
    })

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

        nrof_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'number of trainable parameters: {nrof_params}')

        # TODO: инициализируйте оптимайзер через getattr(torch.optim, self.cfg.optimizer_name)
        self.optimizer = getattr(torch.optim, self.cfg.optimizer_name)(self.model.parameters(), lr=self.cfg.lr)

    def save_model(self, filename):
        """
            Сохранение весов модели с помощью torch.save()
            :param filename: str - название файла
            TODO: реализовать сохранение модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        torch.save(self.model.state_dict(), path)

    def load_model(self, filename):
        """
            Загрузка весов модели с помощью torch.load()
            :param filename: str - название файла
            TODO: реализовать выгрузку весов модели по пути os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        """
        path = os.path.join(self.cfg.exp_dir, f"{filename}.pt")
        self.model.load_state_dict(torch.load(path))

    def make_step(self, batch, update_model=True):
        """
            Этот метод выполняет один шаг обучения, включая forward pass, вычисление целевой функции,
            backward pass и обновление весов модели (если update_model=True).

            :param batch: dict of data with keys ["image", "label"]
            :param update_model: bool - если True, необходимо сделать backward pass и обновить веса модели
            :return: значение функции потерь, выход модели
            # TODO: реализуйте инференс модели для данных batch, посчитайте значение целевой функции
        """
        inputs, labels = batch['image'], batch['label']
        labels = labels.long()

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)

        if update_model:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), logits


    def train_epoch(self, *args, **kwargs):
        """
            Обучение модели на self.train_dataloader в течение одной эпохи. Метод проходит через все обучающие данные и
            вызывает метод self.make_step() на каждом шаге.

            TODO: реализуйте функцию обучения с использованием метода self.make_step(batch, update_model=True),
                залогируйте на каждом шаге значение целевой функции и accuracy на batch
        """
        self.model.train()
        #total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_dataloader):
            loss, logits = self.make_step(batch, update_model=True)

            _, predicted_labels = torch.max(logits, 1)
            accuracy_value = accuracy(predicted_labels, batch['label'])
            self.neptune_logger.save_param(
                'train',
                ['target_function_value', 'accuracy'],
                [loss, accuracy_value]
            )
            #total_loss += loss


    def evaluate(self, *args, **kwargs):
        """
            Метод используется для проверки производительности модели на обучающих/тестовых данных. Сначала модель
            переводится в режим оценки (model.eval()), затем данные последовательно подаются на вход модели, по
            полученным выходам вычисляются метрики производительности, такие как значение целевой функции, accuracy

            TODO: реализуйте функцию оценки с использованием метода self.make_step(batch, update_model=False),
                залогируйте значения целевой функции и accuracy, постройте confusion_matrix
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                loss, logits = self.make_step(batch, update_model=False)

                _, predicted_labels = torch.max(logits.float(), 1)
                total_loss += loss

                all_predictions.extend(predicted_labels.tolist())
                all_labels.extend(batch['label'].tolist())

        #avg_loss = total_loss / len(self.test_dataloader)

        accuracy_value = accuracy(torch.tensor(all_predictions), torch.tensor(all_labels))
        self.neptune_logger.save_param(
            'train/test',
            ['target_function_value', 'accuracy'],
            [total_loss, accuracy_value]
        )
        #print(f"Average Loss: {avg_loss}, Accuracy: {accuracy_value}")

        conf_matrix = confusion_matrix(all_predictions, all_labels)
        self.neptune_logger.save_plot('evaluation', 'confusion_matrix', conf_matrix)
        return accuracy_value

    def fit(self, num_epochs : int):
        """
            Основной цикл обучения модели. Данная функция должна содержать один цикл на заданное количество эпох.
            На каждой эпохе сначала происходит обучение модели на обучающих данных с помощью метода self.train_epoch(),
            а затем оценка производительности модели на тестовых данных с помощью метода self.evaluate()

            # TODO: реализуйте основной цикл обучения модели, сохраните веса модели с лучшим значением accuracy на
                тестовой выборке
        """
        best_accuracy = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")

            self.train_epoch()

            accuracy = self.evaluate()

            print('[{:d}]: accuracy {:.4f}'.format(epoch, accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                self.save_model(f"best_model_epoch_{best_epoch}")

    def overfitting_on_batch(self, max_step=100):
        """
            Оверфиттинг на одном батче. Эта функция может быть полезна для отладки и оценки способности вашей
            модели обучаться и обновлять свои веса в ответ на полученные данные.
        """
        batch = next(iter(self.train_dataloader))
        for step in range(max_step):
            loss, output = self.make_step(batch, update_model=True)
            if step % 10 == 0:
                _, predicted_labels = torch.max(output, 1)
                acc = accuracy(predicted_labels, batch['label'])
                print('[{:d}]: loss - {:.4f}, {:.4f}'.format(step + 1, loss, acc))


if __name__ == '__main__':
    from configs.train_cfg import cfg

    trainer = Trainer(cfg)

    # оверффитинг на одном батче
    trainer.overfitting_on_batch()

    # обучение нейронной сети
    trainer.fit()

    # оценка сети на обучающей/валидационной/тестовой выборке
    trainer.evaluate()
