import functools
import random
import string

import sympy as sp


class Variable:

    def __init__(self, name: str, description: str = "",
                 category: str = "",
                 make_unique: bool = False):

        if make_unique:
            alphabet = string.ascii_lowercase + string.digits
            unique_suffix = '_' + ''.join(random.choices(alphabet, k=8))

            name += unique_suffix

        self.name = name

        self.sympy_symbol = sp.sympify(name)
        self.description = description
        self.category = category

        if category == "":
            print(name)

    def __key(self):
        return (self.name, self.components)

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        dict = {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "symbol": self.sympy_symbol,
        }
        return str(dict)

    @functools.total_ordering
    def __lt__(self, other):
        if self.__str__() <= other.__str__():
            return True
