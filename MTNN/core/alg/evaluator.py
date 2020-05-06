# A basic testing/evaluation class.

import torch


class BaseEvaluator:
    """
    Base Evaluator Attributes
    """
    @staticmethod
    def evaluate(model, dataloader):
        raise NotImplementedError


class CategoricalEvaluator(BaseEvaluator):
    """Test for categorical accuracy."""

    @staticmethod
    def evaluate(model, dataloader):
        """"
        :param model:
        :param dataloader:
        :return:
            correct <int>
            total <int>
        """
        correct = 0
        total = 0

        with torch.no_grad():
            for data in dataloader:
                input_data, labels = data
                outputs = model(input_data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct, total

# TODO: Remove?
class SimpleRegressionEvaluator(BaseEvaluator):
    @staticmethod
    def evaluate_output(model, dataset):
        correct = 0
        total = 0

        print("\n   INPUT  |   TARGET  |   PREDICTION")
        with torch.no_grad():
            for data in dataset:
                input_data, labels = data
                outputs = model(input_data)
                predicted, _ = torch.max(outputs.data, 1)

                printout = "{} | {} | {}".format(input_data, labels.tolist(), predicted.tolist())
                print(printout)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print("Percentage correct {} %".format(correct/total))

        return correct, total

