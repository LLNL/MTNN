# A basic testing/evaluation class.

import torch

class BasicEvaluator():
    """Test for categorical accuracy."""
    
    def __init__(self):
        pass

    def evaluate(self, model, dataset):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataset:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total
    
