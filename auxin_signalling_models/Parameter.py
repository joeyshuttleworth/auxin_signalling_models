from .Variable import Variable


class Parameter(Variable):

    def __init__(self, *args, default_value: float = 1.0, **kwargs):

        super().__init__(*args, **kwargs)

        self.default_value = default_value

    def __key(self):
        return (self.name)

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        dict = {
            "name": self.name,
            "description": self.description,
            "default_value": self.default_value
        }
        return str(list(dict.values()))
