"""MTNN/reader.py
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
            'model': yaml_conf['model'],
            'logger': yaml_conf['logger'],
            'components': yaml_conf['components'],
            'dataset': yaml_conf['dataset']
        }
        return yaml_conf_dict
    except Exception as exc:
        print(f"Error in reading YAML Configuration file {exc}")


class YamlConfig:
    """
    Class to read YAML configuraton files using yaml_to_dict() and return properties.
    """
    def __init__(self, yaml_conf_path: str):
        #self._config = yaml_to_dict(yaml_conf_path)
        self._config = yaml.load(open(yaml_conf_path, "r"), Loader = yaml.SafeLoader)

    def get_property(self, property_name) -> str:
        if property_name not in self._config.keys():
            print( f'{property_name} is not in the configuration file')
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
    def prolongation(self) -> str:
        return self.get_property('prolongation')

    @property
    def data(self) -> str:
        return self.get_property('data')


    @property
    def trainer(self):
        return self.get_property('components')
