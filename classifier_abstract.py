from abc import ABCMeta, abstractmethod


class ClassifierAbstract(metaclass=ABCMeta):
    @property
    @abstractmethod
    def description(self):
        ...

    @abstractmethod
    def classify(self, text):
        ...
