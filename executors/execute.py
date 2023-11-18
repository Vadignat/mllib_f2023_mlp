from configs.train_cfg import cfg
from executors.mlp_trainer import Trainer

trainer = Trainer(cfg)
# оверффитинг на одном батче
trainer.overfitting_on_batch(max_step=100)

# обучение нейронной сети
trainer.fit(10)

# оценка сети на обучающей/валидационной/тестовой выборке
trainer.evaluate()