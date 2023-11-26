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
    
    @staticmethod
    def count_stair_climbs_iter(n: int) -> int:
        if n == 0 or n == 1:
            return 1
        
        prev, curr = 1, 1
        for i in range(2, n + 1):
            temp = curr
            curr = prev + curr
            prev = temp
        
        return curr
    
    @staticmethod
    def count_stair_climbs_dp(n: int) -> int:
        if n == 0 or n == 1:
            return 1

        dp = [0] * (n + 1)
        dp[0] = dp[1] = 1
        
        for i in range(2, n + 1):
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[n]
    
    @staticmethod
    def is_happy(n: int) -> bool: # happy if iterative sums of squares of digits eventually equals 1 (and stays)
        
        def sum_of_digit_squares(n: int) -> int:
            sum = 0
            m = n
            
            while m > 0:
                sum += (m % 10) ** 2
                m //= 10

            return sum
        
        cache = set()
        while (n != 1):
            n = sum_of_digit_squares(n)
            if (n in cache):
                return False
            else:
                cache.add(n)

        return True
    
    @staticmethod
    def is_power_of_two(n: int) -> bool:
        product = 2 ** 0
        power = 1

        while (product <= n):
            if (product == n):
                return True
            product = 2 ** power
            power += 1

        return False
    
    @staticmethod
    def is_power_of_two_opt(n: int) -> bool:
        return bin(n)[2] == "1" and bin(n).count("1") == 1
    
    @staticmethod
    def add_digits_until_single(num: int) -> int:
        while num > 9:
            sum = 0
            while num:
                sum += num % 10
                num //= 10
            num = sum
    
        return num
    
    @staticmethod
    def is_ugly(n: int) -> bool:
        
        if (n <= 0):
            return False

        for i in [2, 3, 5]:
            while (n % i == 0):
                n //= i

        return (n == 1)
    
    @staticmethod
    def reverse_32bit_n(n: int) -> int:
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

n = 2
result = NumUtils.count_stair_climbs_iter(n)
print(f'climb_stairs_iter({n}) = {result}')
assert (result == 2)
n = 3
result = NumUtils.count_stair_climbs_iter(n)
print(f'climb_stairs_iter({n}) = {result}')
assert (result == 3)
n = 4
result = NumUtils.count_stair_climbs_iter(n)
print(f'climb_stairs_iter({n}) = {result}')
assert (result == 5)
n = 45
result = NumUtils.count_stair_climbs_iter(n)
print(f'climb_stairs_iter({n}) = {result}')
assert (result == 1836311903)
n = 2
result = NumUtils.count_stair_climbs_dp(n)
print(f'climb_stairs_dp({n}) = {result}')
assert (result == 2)
n = 3
result = NumUtils.count_stair_climbs_dp(n)
print(f'climb_stairs_dp({n}) = {result}')
assert (result == 3)
n = 4
result = NumUtils.count_stair_climbs_dp(n)
print(f'climb_stairs_dp({n}) = {result}')
assert (result == 5)
n = 45
result = NumUtils.count_stair_climbs_dp(n)
print(f'climb_stairs_dp({n}) = {result}')
assert (result == 1836311903)

n = 1
result = NumUtils.is_happy(n)
print(f'is_happy({n}) = {result}')
assert (result == True)
n = 19
result = NumUtils.is_happy(n)
print(f'is_happy({n}) = {result}')
assert (result == True)
n = 2
result = NumUtils.is_happy(n)
print(f'is_happy({n}) = {result}')
assert (result == False)

n = 1
result = NumUtils.is_power_of_two(n)
print(f'is_power_of_two({n}) = {result}')
assert (result == True)
n = 16
result = NumUtils.is_power_of_two(n)
print(f'is_power_of_two({n}) = {result}')
assert (result == True)
n = 3
result = NumUtils.is_power_of_two(n)
print(f'is_power_of_two({n}) = {result}')
assert (result == False)

n = 1
result = NumUtils.is_power_of_two_opt(n)
print(f'is_power_of_two_opt({n}) = {result}')
assert (result == True)
n = 16
result = NumUtils.is_power_of_two_opt(n)
print(f'is_power_of_two_opt({n}) = {result}')
assert (result == True)
n = 3
result = NumUtils.is_power_of_two_opt(n)
print(f'is_power_of_two_opt({n}) = {result}')
assert (result == False)

n = 38
result = NumUtils.add_digits_until_single(n)
print(f'add_digits_until_single({n}) = {result}')
assert (result == 2)
n = 1
result = NumUtils.add_digits_until_single(n)
print(f'add_digits_until_single({n}) = {result}')
assert (result == 1)
n = 0
result = NumUtils.add_digits_until_single(n)
print(f'add_digits_until_single({n}) = {result}')
assert (result == 0)

n = 0
result = NumUtils.is_ugly(n)
print(f'is_ugly({n}) = {result}')
assert (result == False)
n = 1
result = NumUtils.is_ugly(n)
print(f'is_ugly({n}) = {result}')
assert (result == True)
n = 6
result = NumUtils.is_ugly(n)
print(f'is_ugly({n}) = {result}')
assert (result == True)
n = 14
result = NumUtils.is_ugly(n)
print(f'is_ugly({n}) = {result}')
assert (result == False)

# 00000010100101000001111010011100
n = 14
result = NumUtils.reverse_32bit_n(n)
print(f'reverse_32bit_n({n}) = {result}')
assert (result == 32)
# 00000010100101000001111010011100
n = 14
result = NumUtils.reverse_32bit_n(n)
print(f'reverse_32bit_n({n}) = {result}')
assert (result == 32)
