from abc import ABCMeta, abstractmethod


class ClassifierAbstract(metaclass=ABCMeta):
    """
    This abstract class provides common interface for classifiers
    used in bot. Used for typechecking to ensure classifiers have
    required methods.
    """

    @property
    @abstractmethod
    def description(self):
        """
        :return: string with brief information about model used
        """
        ...

    @abstractmethod
    def classify(self, text):
        """
        :param text: str with text to classify
        :return: object that represents the result of classification
        """
        ...
