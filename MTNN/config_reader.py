"""MTNN/config_reader.py
Reads and Returns configuration parameters
"""
 # third-party
import yaml


class YamlConfig:
    """
    Reads YAML files and returns parameters
    """
    def __init__(self, conf_path: str):
        yaml_conf = yaml.load(open(conf_path, "r"), Loader = yaml.SafeLoader)

        conf = {
            'model_type': yaml_conf['model_type'],
            'input_size': yaml_conf['input_size'],
            'layers': yaml_conf['layers'],
            'hyperparameters': yaml_conf['hyperparameters'],
            'num_epochs': yaml_conf['hyperparameters']['num_epochs'],
            'log_interval': yaml_conf['hyperparameters']['log_interval'],
            'batch_size_train': yaml_conf['hyperparameters']['batch_size_train'],
            'batch_size_test': yaml_conf['hyperparameters']['batch_size_test'],
            'objective': yaml_conf['hyperparameters']['objective'],
            'learning_rate': yaml_conf['hyperparameters']['learning_rate'],
            'momentum': yaml_conf['hyperparameters']['momentum'],
            'optimization': yaml_conf['hyperparameters']['optimization']
        }
        self._config = conf

    def get_property(self, property_name):

        if property_name not in self._config.keys():
            print(KeyError)
        return self._config[property_name]

    @property
    def model_type(self):
        return self.get_property('model_type')

    @property
    def input_size(self):
        return self.get_property('input_size')

    @property
    def layers(self):
        return self.get_property('layers')

    @property
    def hyperparameters(self):
        return self.get_property('hyperparameters')

    @property
    def num_epochs(self):
        return self.get_property('num_epochs')

    @property
    def log_intervals(self):
        return self.get_property('log_intervals')

    @property
    def batch_size_train(self):
        return self.get_property('batch_size_train')

    @property
    def batch_size_test(self):
        return self.get_property('batch_size_test')

    @property
    def objective(self):
        return self.get_property('objective')

    @property
    def learning_rate(self):
        return self.get_property('learning_rate')

    @property
    def momentum(self):
        return self.get_property('momentum')

    @property
    def optimization(self):
        return self.get_property('optimization')
