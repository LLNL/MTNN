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
                input_data, labels = data
                outputs = model(input_data)
                _, predicted = torch.max(outputs.data, 1)


                print("Prediction", predicted.data.item(), input_data.data.item(), labels.data.item())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total

    def evaluate_output(self, model, dataset):
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataset:
                input_data, labels = data
                outputs = model(input_data)
                _, predicted = torch.max(outputs.data, 1)

                printout = "Correct {} Prediction{}".format(labels.tolist(), predicted.tolist())
                print(printout)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Percentage correct {} %".format(correct/total))

        return correct, total

