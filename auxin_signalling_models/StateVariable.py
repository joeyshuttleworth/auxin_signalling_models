from typing import Iterable

from .Variable import Variable


class StateVariable(Variable):

    def __init__(self, *args,
                 initial_condition: float = 1.0,
                 components: Iterable = [],
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.components = tuple(components)
        self.initial_condition = initial_condition

    def __key(self):
        return (self.name, self.components)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        dict = {"name": self.name,
                "description": self.description,
                "components": self.components,
                "category": self.category,
                "initial_condition": self.initial_condition
                }

        return str(dict)

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        # Define the default sorting order based on price
        return str(self) < str(other)
