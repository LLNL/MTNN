"""MTNN/config_reader.py
Reads and Returns configuration parameters
"""
 # third-party
import yaml


def yaml_to_dict(yaml_conf_path: str):
    """
    Takes absolute or relative YAML configuration path <str> and parses YAML into a dictionary
    Args:
        yaml_conf_path <str>: absolute or relative path to YAML configuration file.

    Returns:
        yaml_conf_dict <dict>: dictionary of configuration properties parsed from YAML file
    """
    try:
        yaml_conf = yaml.load(open(yaml_conf_path, "r"), Loader = yaml.SafeLoader)

        yaml_conf_dict = {
            # Neural Network Architecture
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
            'optimization': yaml_conf['hyperparameters']['optimization'],

            # Multigrid scheme
            'multigrid_scheme': yaml_conf['multigrid_scheme'],
            'prolongation': yaml_conf['multigrid_scheme']['prolongation'],
            'restriction': yaml_conf['multigrid_scheme']['restriction'],

            # Dataset
            'datset': yaml_conf['dataset']
        }
    except FileNotFoundError as exc:
        print(exc)

    return yaml_conf_dict


class YamlConfig:
    """
    Class to read YAML configuraton files using yaml_to_dict() and return properties.
    """
    def __init__(self, conf_path: str):
        self._config = yaml_to_dict(conf_path)

    def get_property(self, property_name) -> str:
        if property_name not in self._config.keys():
            print(KeyError)
        return self._config[property_name]

    @property
    def model_type(self) -> str:
        return self.get_property('model_type')

    @property
    def input_size(self) -> str:
        return self.get_property('input_size')

    @property
    def layers(self) -> str:
        return self.get_property('layers')

    @property
    def hyperparameters(self) -> str:
        return self.get_property('hyperparameters')

    @property
    def num_epochs(self) -> str:
        return self.get_property('num_epochs')

    @property
    def log_intervals(self) -> str:
        return self.get_property('log_intervals')

    @property
    def batch_size_train(self) -> str:
        return self.get_property('batch_size_train')

    @property
    def batch_size_test(self) -> str:
        return self.get_property('batch_size_test')

    @property
    def objective(self) -> str:
        return self.get_property('objective')

    @property
    def learning_rate(self) -> str:
        return self.get_property('learning_rate')

    @property
    def momentum(self) -> str:
        return self.get_property('momentum')

    @property
    def optimization(self) -> str:
        return self.get_property('optimization')

    @property
    def multigrid_scheme(self) -> str:
        return self.get_property('multigrid_scheme')

    @property
    def prolongation(self) -> str:
        return self.get_property('prolongation')

    @property
    def data(self) -> str:
        return self.get_property('data')
