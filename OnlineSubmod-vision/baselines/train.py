from cords import train_sl
from cords.utils.config_utils import load_config_data

config_file = '/content/cords/configs/SL/config_glister_cifar10.py'
cfg = load_config_data(config_file)
clf = train_sl.TrainClassifier(cfg)
clf.train()