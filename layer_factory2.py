
class MakeLayers:
    """
    MakeLayers interface declares the methods implemented according to model-type
    """
    def __init__(self, config_file):
        self.config = config_file

    def createLayers(self):
        if self.config["model-type"] == "fully-connected":
            self.layer_dict = (self.config)
        else:
            self.layer_dict = None

