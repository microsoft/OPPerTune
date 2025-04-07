import abc


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def get_step_value(self, grad):
        pass
