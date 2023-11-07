from math import floor

class NumUtils:

    @staticmethod
    def my_sqrt(x: int) -> int:
        return floor(x ** 0.5)

result = NumUtils.my_sqrt(4)
assert (result == 2)
result = NumUtils.my_sqrt(5)
assert (result == 2)
result = NumUtils.my_sqrt(2147483647)
assert (result == 46340)
