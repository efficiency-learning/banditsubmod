from cords.utils.config_utils import load_config_data

# config_file = './cords/configs/SL/config_gradmatch_cifar10.py'
config_file = './cords/configs/SL/custom_configs_glister_pb/config_glister_mnist_frac_0.5.py'
cfg = load_config_data(config_file)
print(cfg)
from train_sl import TrainClassifier
clf = TrainClassifier(cfg)
clf.train()