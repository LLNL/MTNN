"""
Builds Trainer class
"""


# local

def build(configuration_path: str):
    myYaml = config_reader.YamlConfig(configuration_path)
    trainer(myYaml)
    return self
