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
    def can_construct_ransom_note(ransom_note: str, magazine: str) -> bool:
        
        for letter in ransom_note:
            mag_i = magazine.find(letter)
            if (mag_i == -1):
                return False
            else:
                magazine = magazine.replace(letter, '_', 1)

        return True
    
    @staticmethod
    def first_uniq_char(s: str) -> int:
        unique_chars = {}

        for i, letter in enumerate(s):
            if (letter in unique_chars):
                unique_chars[letter] = -1
            else:
                unique_chars[letter] = i

        for index in unique_chars.values():
            if (index >= 0):
                return index
            
        return -1

    @staticmethod
    def find_the_diff_letter(s: str, t: str) -> str:
        ascii_sum = 0

        for c in s:
            ascii_sum += ord(c)

        for c in t:
            ascii_sum -= ord(c)

        return chr(abs(ascii_sum))
    
    @staticmethod
    def is_subsequence(s: str, t: str) -> bool:
        si = 0
        for c in t:
            if (si < len(s) and c == s[si]):
                si += 1

        return si == len(s)
    
    @staticmethod
    def read_binary_watch(bin_nums_turned_on: int) -> List[str]:
        nums_with_n_bin_ones = []

        def count_ones(num_str: str) -> int:
            ones_count = 0
            padded_num_str = pad_with_zeros(num_str, 4)
            hr_str = padded_num_str[:2]
            mn_str = padded_num_str[2:]
            bin_num_str = str(bin(int(hr_str))[2:]) + str(bin(int(mn_str))[2:])
            for bit in bin_num_str:
                if (bit == '1'):
                    ones_count += 1
            if (len(hr_str) > 1 and hr_str.startswith('0')):
                hr_str = hr_str[1:]
            return (ones_count, f'{hr_str}:{mn_str}')

        def pad_with_zeros(num_str: str, length: int) -> str:
            if (len(num_str) < length):
                num_str = num_str.zfill(length)
            return num_str

        for i in range(1200):
            if (i % 100 < 60):
                time_str = str(i)
                ones_count, formatted_time_str = count_ones(time_str)
                if (ones_count == bin_nums_turned_on):
                    nums_with_n_bin_ones.append(formatted_time_str)

        return nums_with_n_bin_ones
    
    @staticmethod
    def longest_palindrome_len(s: str) -> int:
        long_palin_len = 0
        letter_counts = {}

        for c in s:
            if (c in letter_counts):
                letter_counts[c] += 1
            else:
                letter_counts[c] = 1

        add_one = False
        for v in letter_counts.values():
            long_palin_len += v - (v % 2)
            if (v % 2 > 0):
                add_one = True

        if (add_one):
            long_palin_len += 1

        return long_palin_len

    @staticmethod
    def add_int_strings(num1: str, num2: str) -> str:
        num_strings = (num1, num2) if (len(num1) >= len(num2)) else (num2, num1)
        num_strings = (num_strings[0], num_strings[1].zfill(len(num_strings[0])))

        digits = {}
        for i in range(0, 10):
            digits[str(i)] = i

        result = ''
        carry = False
        for i in range(len(num_strings[0]) - 1, -1, -1):
            digit_sum = digits[num_strings[0][i]] + digits[num_strings[1][i]]
            if (carry):
                digit_sum += 1
                carry = False
            if (digit_sum > 9):
                digit_sum %= 10
                carry = True
            result = f'{digit_sum}{result}'

        if (carry):
            result = f'1{result}'

        return result
    
    @staticmethod
    def is_repeated_substring_pattern(s: str) -> bool:
        return s in (s + s)[1:-1]
    
    @staticmethod
    def correct_capitalization(word: str) -> bool:
        if (len(word) == 1):
            return True
        
        expect_uppers = True if (word[0].isupper() and word[1].isupper()) else False
        for i in range(1, len(word)):
            letter = word[i]
            if (not expect_uppers == letter.isupper()):
                return False
        
        return True
    

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

assert_string_parsing(StringParsing.can_construct_ransom_note, ["a", "b"], False)
assert_string_parsing(StringParsing.can_construct_ransom_note, ["aa", "ab"], False)
assert_string_parsing(StringParsing.can_construct_ransom_note, ["aa", "aab"], True)

assert_string_parsing(StringParsing.first_uniq_char, ["leetcode"], 0)
assert_string_parsing(StringParsing.first_uniq_char, ["loveleetcode"], 2)
assert_string_parsing(StringParsing.first_uniq_char, ["aabb"], -1)

assert_string_parsing(StringParsing.find_the_diff_letter, ["abcd", "abcde"], "e")
assert_string_parsing(StringParsing.find_the_diff_letter, ["", "y"], "y")
assert_string_parsing(StringParsing.find_the_diff_letter, ["asdf", "asdff"], "f")

assert_string_parsing(StringParsing.is_subsequence, ["abc", "ahbgdc"], True)
assert_string_parsing(StringParsing.is_subsequence, ["axc", "ahbgdc"], False)
assert_string_parsing(StringParsing.is_subsequence, ["aba", "ahcagbacb"], True)

assert_string_parsing(StringParsing.read_binary_watch, [1], ["0:01", "0:02", "0:04", "0:08", "0:16", "0:32", "1:00", "2:00", "4:00", "8:00"])
assert_string_parsing(StringParsing.read_binary_watch, [9], [])
assert_string_parsing(StringParsing.read_binary_watch, [2], ['0:03', '0:05', '0:06', '0:09', '0:10', '0:12', '0:17', '0:18', '0:20', '0:24', '0:33', '0:34', '0:36', '0:40', '0:48', '1:01', '1:02', '1:04', '1:08', '1:16', '1:32', '2:01', '2:02', '2:04', '2:08', '2:16', '2:32', '3:00', '4:01', '4:02', '4:04', '4:08', '4:16', '4:32', '5:00', '6:00', '8:01', '8:02', '8:04', '8:08', '8:16', '8:32', '9:00', '10:00'])

assert_string_parsing(StringParsing.longest_palindrome_len, ["abccccdd"], 7)
assert_string_parsing(StringParsing.longest_palindrome_len, ["a"], 1)

assert_string_parsing(StringParsing.add_int_strings, ["11", "123"], "134")
assert_string_parsing(StringParsing.add_int_strings, ["456", "77"], "533")
assert_string_parsing(StringParsing.add_int_strings, ["0", "0"], "0")

assert_string_parsing(StringParsing.is_repeated_substring_pattern, ["abab"], True)
assert_string_parsing(StringParsing.is_repeated_substring_pattern, ["aba"], False)
assert_string_parsing(StringParsing.is_repeated_substring_pattern, ["abcabcabcabc"], True)

assert_string_parsing(StringParsing.correct_capitalization, ["USA"], True)
assert_string_parsing(StringParsing.correct_capitalization, ["FlaG"], False)
