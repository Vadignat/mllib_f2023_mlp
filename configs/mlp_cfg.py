from easydict import EasyDict
from utils.enums import InitWeightType

cfg = EasyDict()

cfg.layers = [
    ('Linear', {'in_features': 28 * 28, 'out_features': 100}),
    ('ReLU', {}),
    ('Linear', {'in_features': 100, 'out_features': 10}),
]

cfg.init_type = InitWeightType.xavier_normal_.name

#cfg.init_type = InitWeightType.xavier_uniform_.name