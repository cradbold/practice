class NumUtils:

    @staticmethod
    def my_sqrt(x: int) -> int:
        return x ** 0.5

result = NumUtils.my_sqrt(4)
assert (result == 2)
