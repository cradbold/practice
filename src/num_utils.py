from math import floor

class NumUtils:

    @staticmethod
    def my_sqrt(x: int) -> int:
        if (x == 0):
            return 0

        if (x <= 3):
            return 1
        
        if (x == 4):
            return 2

        l, r = 2, x // 2
        while (l <= r):
            m = (l + r) // 2
            sq = m * m
            if (sq < x):
                l = m + 1
            elif (sq > x):
                r = m - 1
            else:
                return m
            
        return r
    
    # You are climbing a staircase. It takes n steps to reach the top.
    # Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    @staticmethod
    def climbStairs(n: int) -> int:
        pass
    

x = 0
result = NumUtils.my_sqrt(x)
print(f'sqrt({x}) = {result}')
assert (result == 0)
x = 2
result = NumUtils.my_sqrt(x)
print(f'sqrt({x}) = {result}')
assert (result == 1)
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
