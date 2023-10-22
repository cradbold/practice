from typing import Callable, List, Any

class StringParsing:

    @staticmethod
    def longest_common_prefix(strs: List[str]) -> str:
        prefix = strs[0]

        for str in strs[1:]:
            while (str.find(prefix)) != 0:
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        
        return prefix

    @staticmethod
    def roman_to_int(s: str) -> int:
        sum = 0

        i = 0
        while (i < len(s)):
            c = s[i]

            match c:
                case "I":
                    if (i + 1 < len(s)):
                        next = s[i + 1]
                        if (next == "V"):
                            sum += 4
                            i += 2
                            continue
                        elif (next == "X"):
                            sum += 9
                            i += 2
                            continue
                    sum += 1
                case "V":
                    sum += 5
                case "X":
                    if (i + 1 < len(s)):
                        next = s[i + 1]
                        if (next == "L"):
                            sum += 40
                            i += 2
                            continue
                        elif (next == "C"):
                            sum += 90
                            i += 2
                            continue
                    sum += 10
                case "L":
                    sum += 50
                case "C":
                    if (i + 1 < len(s)):
                        next = s[i + 1]
                        if (next == "D"):
                            sum += 400
                            i += 2
                            continue
                        elif (next == "M"):
                            sum += 900
                            i += 2
                            continue
                    sum += 100
                case "D":
                    sum += 500
                case "M":
                    sum += 1000
                case _:
                    print(f"Unrecognized char: {c}")

            i += 1

        return sum

def assert_string_parsing(func: Callable, args: List, val: Any) -> None:
    print(f'Calling {func.__name__} with args: {args} and asserting return value: {val}')
    result = func(*args)
    print(f'  Result: {result == val}')
    assert result == val

assert_string_parsing(StringParsing.longest_common_prefix, [["flower","flow","flight"]], "fl")
assert_string_parsing(StringParsing.longest_common_prefix, [["dog","racecar","car"]], "")

assert_string_parsing(StringParsing.roman_to_int, ["III"], 3)
assert_string_parsing(StringParsing.roman_to_int, ["LVIII"], 58)
assert_string_parsing(StringParsing.roman_to_int, ["MCMXCIV"], 1994)
