from math import floor

class NumUtils:

    @staticmethod
    def my_sqrt(x: int) -> int:

        if (x <= 1):
            return x
        
        for val in range(1, x):
            sq = val * val
            if (sq > x):
                return val - 1

        return -1
    

x = 4
result = NumUtils.my_sqrt(x)
print(f'sqrt({x}) = {result}')
assert (result == 2)
x = 5
result = NumUtils.my_sqrt(x)
print(f'sqrt({x}) = {result}')
assert (result == 2)
x = 8
result = NumUtils.my_sqrt(x)
print(f'sqrt({x}) = {result}')
assert (result == 2)
x = 2147483647
result = NumUtils.my_sqrt(x)
print(f'sqrt({x}) = {result}')
assert (result == 46340)
