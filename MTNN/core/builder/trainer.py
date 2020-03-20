"""
Builds Trainer class
"""


# local
import MTNN.core.components

def build(configuration_path: str):
    myYaml = config_reader.YamlConfig(configuration_path)
    trainer(myYaml)
    return self
