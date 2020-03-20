"""
Trainer class
"""
class Trainer:
    def __init__(self, myYaml):
        self.num_epochs = myYaml.get_property('num_epochs')
        self.batch_size_train = myYaml.get_property('batch_size_train')
        self.batch_size_test = myYaml.get_property('batch_size_test')
        self.optimizer = myYaml.get_property('optimizer')

