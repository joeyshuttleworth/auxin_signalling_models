import random
import string
from typing import Iterable

import sympy as sp


class Variable:

    def __init__(self, name: str, description: str = "",
                 make_unique: bool = False,
                 components: Iterable = []):
        if make_unique:
            alphabet = string.ascii_lowercase + string.digits
            unique_suffix = '_' + ''.join(random.choices(alphabet, k=8))

            name += unique_suffix

        self.name = name

        self.sympy_symbol = sp.sympify(name)
        self.description = description

    def __key(self):
        return (self.name, self.components)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        dict = {"name": self.name,
                "description": self.description,
                "components": self.components}
        return str(dict)
