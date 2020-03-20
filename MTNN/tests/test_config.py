# Filename: MTNN/tests/test_config.py
# Unit test for MTNN/generator.py
# Tests and validates Yaml correctness for Yaml model configuration files

import pytest
import yaml

# Collector
def pytest_collect_file(parent, filepath):
    if filepath.ext == ".yml" and filepath.basename.startsiwth("test"):
        return yaml.YamlFile(filepath, parent)