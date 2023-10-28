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
    
    @staticmethod
    def has_valid_groupings(s: str) -> bool:
        stack = []

        for c in s:
            match c:
                case "(" | "{" | "[":
                    stack.insert(0, c)
                case ")" | "}" | "]":
                    opening_c = StringParsing.get_opening_char(c)
                    if (len(stack) > 0 and stack[0] == opening_c):
                        stack.pop(0)
                    else:
                        return False
                case _:
                    print(f"Unrecognized char: {c}")

        if (len(stack)):
            return False
        else:
            return True
        
    @staticmethod
    def get_opening_char(s: str) -> str:
        match s:
            case ")":
                return "("
            case "}":
                return "{"
            case "]":
                return "["
            case _:
                print(f"Unsupported string: {s}")

    @staticmethod
    def search_str(haystack: str, needle: str) -> int:
        for i in range(len(haystack) - len(needle) + 1):

            hj = i
            nj = 0
            while(haystack[hj] == needle[nj]):
                if (nj == len(needle) - 1):
                    return i
                else:
                    hj += 1
                    nj += 1

        return -1


def assert_string_parsing(func: Callable, args: List, val: Any) -> None:
    print(f'Calling {func.__name__} with args: {args} and asserting return value: {val}')
    result = func(*args)
    print(f'  Result: {result == val}')
    assert result == val

assert_string_parsing(StringParsing.longest_common_prefix, [["flower", "flow", "flight"]], "fl")
assert_string_parsing(StringParsing.longest_common_prefix, [["dog", "racecar", "car"]], "")

assert_string_parsing(StringParsing.roman_to_int, ["III"], 3)
assert_string_parsing(StringParsing.roman_to_int, ["LVIII"], 58)
assert_string_parsing(StringParsing.roman_to_int, ["MCMXCIV"], 1994)

assert_string_parsing(StringParsing.get_opening_char, [")"], "(")
assert_string_parsing(StringParsing.get_opening_char, ["}"], "{")
assert_string_parsing(StringParsing.get_opening_char, ["]"], "[")

assert_string_parsing(StringParsing.has_valid_groupings, ["()"], True)
assert_string_parsing(StringParsing.has_valid_groupings, ["()[]{}"], True)
assert_string_parsing(StringParsing.has_valid_groupings, ["(]"], False)

assert_string_parsing(StringParsing.search_str, ["sadbutsad", "sad"], 0)
assert_string_parsing(StringParsing.search_str, ["leetcode", "leeto"], -1)
