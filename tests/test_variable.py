from auxin_signalling_models.Variable import Variable
import pytest


class TestVariable:
    def test_string(self):
        var = Variable("x", "a variable")
        var2 = Variable("x", "another variable", make_unique=True)
        var3 = Variable("x", "another variable", make_unique=True)

        assert str(var) != str(var2) and str(var2) != str(var3) \
            and str(var) != str(var3)

        assert repr(var) != repr(var2) and repr(var2) != repr(var3) \
            and repr(var) != repr(var3)

        assert var != var2 and var2 != var3 and var != var3
