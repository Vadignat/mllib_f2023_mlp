import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()

        self.cfg = cfg

        self.layers = self._init_layers()
        self.apply(self._init_weights)

    def _init_layers(self):
        """
            Инициализация слоев нейронной сети.

            Описание необходимых слоев задается в self.cfg.layers в следующем виде:
                 [
                    (name_1, params_1),
                    ...,
                    (name_N, params_N)
                ],
                где name_i (str) - название класса из nn (Linear, ReLU и т.д.), params_i (dict) - параметры этого слоя
            :return: список инициализированных слоев

            TODO: необходимо инициализировать слои, заданные с помощью self.cfg.layers, данная функция должна быть
                универсальной (описание модели вынесено в конфиг, основной код не должен меняться в зависимости от
                эксперимента), можно использовать getattr(nn, name) и nn.Sequential/nn.ModuleList
        """
        layers = []
        for name, params in self.cfg.layers:
            layer = getattr(nn, name)(**params)
            layers.append(layer)
        return nn.Sequential(*layers)

    @torch.no_grad()
    def _init_weights(self, m):
        """
            Инициализация параметров линейный слоев согласно заданному типу self.cfg.init_type.
            # TODO: реализуйте этот метод, можно использовать getattr(nn.init, self.cfg.init_type)
        """
        if isinstance(m, nn.Linear):
            getattr(nn.init, self.cfg.init_type)(m.weight)

    def forward(self, inputs):
        """
            Forward pass нейронной сети, все вычисления производятся для батча
            :param inputs: torch.Tensor(batch_size, height, weight)
            :return output of the model: torch.Tensor(batch_size, nrof_classes)

            TODO: реализуйте этот метод
        """
        x = inputs.view(inputs.size(0), -1)
        output = self.layers(x)

        return output


if __name__ == '__main__':
    from configs.mlp_cfg import cfg

    model = MLP(cfg)

    # TODO: вывести количество обучаемых параметров нейронной сети
    nrof_params = None
    print(f'number of trainable parameters: {nrof_params}')