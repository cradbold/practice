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
            while (haystack[hj] == needle[nj]):
                if (nj == len(needle) - 1):
                    return i
                else:
                    hj += 1
                    nj += 1

        return -1
    
    @staticmethod
    def last_word_length(s: str) -> int:
        start_i, end_i = 0, len(s)
        found = False

        for i in range(len(s) - 1, -1, -1):
            char = s[i]
        
            if (char != ' ' and not found):
                end_i = i + 1
                found = True
            elif (char == ' ' and found):
                start_i = i + 1
                break

        return end_i - start_i
    
    @staticmethod
    def add_binary_str(a: str, b: str) -> str:
        result = ""
        
        carry = 0
        ai, bi = len(a) - 1, len(b) - 1
        while ai >= 0 or bi >= 0 or carry != 0:
            sum = carry
            if ai >= 0:
                sum += int(a[ai])
                ai -= 1
            if bi >= 0:
                sum += int(b[bi])
                bi -= 1
            result = f'{str(sum % 2)}{result}'
            carry = sum // 2
        
        return result
    
    @staticmethod
    def is_anagram(s: str, t: str) -> bool:
        s_letters = {}

        for c in s:
            if (c in s_letters):
                s_letters[c] += 1
            else:
                s_letters[c] = 1

        for c in t:
            if (c in s_letters):
                s_letters[c] -= 1
            else:
                return False
            
        for count in s_letters.values():
            if (count != 0):
                return False
            
        return True
    
    @staticmethod
    def remove_non_alnum(s: str) -> str:
        result = ""
        for c in s:
            if (c.isalnum()):
                result += c
        return result
    
    @staticmethod
    def is_palindrome_opt(s: str) -> bool:
        li, ri = 0, len(s) - 1

        while (li < ri):
            left, right = s[li].lower(), s[ri].lower()

            if (not left.isalnum()):
                li += 1
            elif (not right.isalnum()):
                ri -= 1
            elif (left != right):
                return False
            else:
                li += 1
                ri -= 1

        return True
    
    @staticmethod
    def is_vowel(s: str) -> bool:
        vowels = ['a', 'e', 'i', 'o', 'u']
        return s.lower() in vowels
    
    @staticmethod
    def reverse_vowels(s: str) -> str:
        result = ""
        vowels = []

        for c in s:
            if (StringParsing.is_vowel(c)):
                vowels.append(c)

        for c in s:
            if (StringParsing.is_vowel(c)):
                result += vowels.pop()
            else:
                result += c

        return result
    
    @staticmethod
    def reverse_vowels(s: str) -> str:
        result = ""
        vowels = []

        for c in s:
            if (StringParsing.is_vowel(c)):
                vowels.append(c)

        for c in s:
            if (StringParsing.is_vowel(c)):
                result += vowels.pop()
            else:
                result += c

        return result
    
    @staticmethod
    def canConstruct(ransomNote: str, magazine: str) -> bool:
        pass


def assert_string_parsing(func: Callable, args: List, val: Any) -> None:
    print(f'Calling {func.__name__} with args: {args} and asserting return value: {val}')
    result = func(*args)
    print(f'  Expected result found: {result == val}')
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

assert_string_parsing(StringParsing.last_word_length, ["Hello World"], 5)
assert_string_parsing(StringParsing.last_word_length, ["   fly me   to   the moon  "], 4)
assert_string_parsing(StringParsing.last_word_length, ["luffy is still joyboy"], 6)
assert_string_parsing(StringParsing.last_word_length, ["a"], 1)
assert_string_parsing(StringParsing.last_word_length, [""], 0)

assert_string_parsing(StringParsing.add_binary_str, ["0", "0"], "0")
assert_string_parsing(StringParsing.add_binary_str, ["1", "0"], "1")
assert_string_parsing(StringParsing.add_binary_str, ["11", "1"], "100")
assert_string_parsing(StringParsing.add_binary_str, ["1010", "1011"], "10101")
assert_string_parsing(StringParsing.add_binary_str, ["10", "11"], "101")

assert_string_parsing(StringParsing.is_anagram, ["anagram", "nagaram"], True)
assert_string_parsing(StringParsing.is_anagram, ["rat", "car"], False)
assert_string_parsing(StringParsing.is_anagram, ["scarab", "cbrass"], False)

assert_string_parsing(StringParsing.remove_non_alnum, ["A man, a plan, a canal: Panama"], "AmanaplanacanalPanama")
assert_string_parsing(StringParsing.remove_non_alnum, ["race a car"], "raceacar")
assert_string_parsing(StringParsing.remove_non_alnum, [" "], "")
assert_string_parsing(StringParsing.remove_non_alnum, [";:"], "")

assert_string_parsing(StringParsing.is_palindrome_opt, ["A man, a plan, a canal: Panama"], True)
assert_string_parsing(StringParsing.is_palindrome_opt, ["race a car"], False)
assert_string_parsing(StringParsing.is_palindrome_opt, [" "], True)

assert_string_parsing(StringParsing.reverse_vowels, ["hello"], "holle")
assert_string_parsing(StringParsing.reverse_vowels, ["leetcode"], "leotcede")
