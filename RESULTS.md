- Базовая модель с lr = 0.1 и batch_size = 32 при 10 эпохах добилась accuracy = 0.8990 
- При добавлении аугментации (Pad + RandomCrop) обучающих данных модель теряет качество: accuracy = 0.3809
- Обучение разных моделей:
1) Метод инициализции весов xavier_uniform - accuracy = 0.8925
2) Модель со следующей архитектурой:
```
cfg.layers = [
    ('Linear', {'in_features': 28 * 28, 'out_features': 200}),
    ('ReLU', {}),
    ('Linear', {'in_features': 200, 'out_features': 128}),
    ('ReLU', {}),
    ('Linear', {'in_features': 128, 'out_features': 64}),
    ('ReLU', {}),
    ('Linear', {'in_features': 64, 'out_features': 10})
]
```
accuracy = 0.9071
3) Модель с LeakyRelu: accuracy = 0.8928
4) Модель с out_features первого слоя = 100:  accuracy = 0.8851
5) Optimizer Adam: accuracy 0.1003
Вывод: среди рассмотренных моделей с наилучшим качеством является 2 модель, но у нее 191634 обучаемых параметров по сравнению с 79510 у базовой модели