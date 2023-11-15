import os
import pickle
import gzip
import numpy as np
from urllib import request


from torch.utils.data import Dataset


class KMNIST(Dataset):
    def __init__(self, cfg, dataset_type: str, transforms=None):
        """
        :param cfg: EasyDict - конфиг
        :param dataset_type: str - тип данных, может принимать значения ['train', 'test']
        """
        self.cfg = cfg
        self.dataset_type = dataset_type
        self.transforms = transforms

        self.nrof_classes = self.cfg.nrof_classes

        self.images, self.labels = [], []
        self._read_dataset()

    def __len__(self):
        """
            Функция __len__ возвращает количество элементов в наборе данных.
            TODO: Реализуйте этот метод
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
            Функция __getitem__ возвращает элемент из набора данных по заданному индексу idx.

            :param idx: int - представляет индекс элемента, к которому вы пытаетесь получить доступ из набора данных
            :return: dict - словарь с двумя ключами: "image" и "label". Ключ "image" соответствует изображению, а ключ
            "label" соответствует метке этого изображения.
            TODO: Реализуйте этот метод, исходное изображение необходимо привести к типу np.float32 и нормализовать
                по заданным self.mean и self.std
        """
        raise NotImplementedError

    def show_statistics(self):
        """
            TODO: Необходимо посчитать количество элементов в наборе данных, количество элементов в каждом классе, а так
                же посчитать среднее (mean) и стандартное отклонение (std) по всем изображениям набора данных
                Результат работы функции вывести в консоль (print())
        """
        raise NotImplementedError

    def _read_dataset(self):
        if not os.path.exists(os.path.join(self.cfg.path, self.cfg.filename)):
            self._download_dataset()
        # считывание данных из pickle файлов, каждое изображение хранится в виде матрицы размера 28х28
        self.dataset = {}
        with open(os.path.join(self.cfg.path, f"{self.dataset_type}_{self.cfg.filename}"), "rb") as f:
            data = pickle.load(f, encoding="latin-1")
        self.images, self.labels = data['images'], data['labels']

    def _download_dataset(self):
        os.makedirs(self.cfg.path, exist_ok=True)
        for name in self.cfg.raw_filename:
            filename = f"{name[0].split('_')[0]}_{self.cfg.filename}"
            if not os.path.exists(os.path.join(self.cfg.path, filename)):
                print("Downloading " + name[1] + "...")
                request.urlretrieve(self.cfg.base_url + name[1], self.cfg.path + name[1])
        self._save_mnist()

    def _save_mnist(self):
        mnist = {'train': {}, 'test': {}}
        for name in self.cfg.raw_filename[:2]:
            data_type = name[0].split('_')[0]
            with gzip.open(self.cfg.path + name[1], 'rb') as f:
                mnist[data_type]['images'] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        for name in self.cfg.raw_filename[-2:]:
            data_type = name[0].split('_')[0]
            with gzip.open(self.cfg.path + name[1], 'rb') as f:
                mnist[data_type]['labels'] = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)
        for data_type in mnist.keys():
            with open(os.path.join(self.cfg.path, f"{data_type}_{self.cfg.filename}"), 'wb') as f:
                pickle.dump(mnist[data_type], f)
        print("Save complete.")


if __name__ == '__main__':
    from configs.kmnist_cfg import cfg

    data = KMNIST(cfg, dataset_type='test')
