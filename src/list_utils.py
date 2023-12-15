from typing import Optional, Callable, List, Any
from collections import deque, Counter
from collections.abc import Iterable
import math

class ListNode:

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Queue:

    def __init__(self):
        self.__deque = deque()

    def enqueue(self, x: int) -> None:
        self.__deque.append(x)

    def dequeue(self) -> int:
        return self.__deque.popleft()

    def peek(self) -> int:
        return self.__deque[0]

    def size(self) -> int:
        return len(self.__deque)
    
    def is_empty(self) -> bool:
        return self.size() == 0

class Stack:

    def __init__(self):
        self.__list = []

    def push(self, x: int) -> None:
        self.__list.append(x)

    def pop(self) -> int:
        return self.__list.pop()

    def peek(self) -> int:
        return self.__list[-1]

    def size(self) -> int:
        return len(self.__list)
    
    def is_empty(self) -> bool:
        return self.size() == 0

class QueueStack:

    def __init__(self):
        self.__queue = Queue()

    def push(self, x: int) -> None:
        if (self.__queue.is_empty()):
            self.__queue.enqueue(x)
        else:
            helper_queue = Queue()
            while (not self.__queue.is_empty()) :
                helper_queue.enqueue(self.__queue.dequeue())
            self.__queue.enqueue(x)
            while (not helper_queue.is_empty()):
                self.__queue.enqueue(helper_queue.dequeue())
            

    def pop(self) -> int:
        result = math.nan

        if (not self.__queue.is_empty()):
            result = self.__queue.dequeue()              

        return result

    def top(self) -> int:
        return self.__queue.peek()

    def empty(self) -> bool:
        return self.__queue.is_empty()
    
class StackQueue:

    def __init__(self):
        self.__stack = Stack()

    def enqueue(self, x: int) -> None:
        if (self.__stack.is_empty()):
            self.__stack.push(x)
        else:
            helper_stack = Stack()
            while (not self.__stack.is_empty()) :
                helper_stack.push(self.__stack.pop())
            self.__stack.push(x)
            while (not helper_stack.is_empty()):
                self.__stack.push(helper_stack.pop())
            
    def dequeue(self) -> int:
        result = math.nan

        if (not self.__stack.is_empty()):
            result = self.__stack.pop()

        return result

    def peek(self) -> int:
        return self.__stack.peek()

    def is_empty(self) -> bool:
        return self.__stack.is_empty()

class ListUtils:

    @staticmethod
    def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = cur = ListNode()
        
        while (list1 and list2):
            if (list1.val < list2.val):
                cur.next = list1
                cur = list1
                list1 = list1.next
            else:
                cur.next = list2
                cur = list2
                list2 = list2.next

        if (list1):
            cur.next = list1

        if (list2):
            cur.next = list2
            
        return head.next
    
    @staticmethod
    def remove_duplicates(nums: List[int]) -> int:
        replace_index = 1

        for i in range(1, len(nums)):
            if (nums[i] != nums[i - 1]):
                nums[replace_index] = nums[i]
                replace_index += 1

        return replace_index
    
    @staticmethod
    def remove_element(nums: List[int], val: int) -> int:
        replace_index = 0

        for i in range(len(nums)):
            if (nums[i] != val):
                nums[replace_index] = nums[i]
                replace_index += 1

        return replace_index
    
    @staticmethod
    def search_insert(nums: List[int], target: int) -> int:
        li = 0
        ri = len(nums) - 1

        while (li <= ri):
            mid = (li + ri) // 2
            if (nums[mid] < target):
                li = mid + 1
            elif (nums[mid] > target):
                ri = mid - 1
            else:
                return mid
        
        return li
    
    @staticmethod
    def plus_one(digits: List[int]) -> List[int]:
        for i in range(len(digits) - 1, -1, -1):
            if digits[i] == 9:
                digits[i] = 0
            else:
                digits[i] = digits[i] + 1
                return digits
                
        return [1] + digits
    
    @staticmethod
    def delete_duplicates(head: Optional[ListNode]) -> Optional[ListNode]:
        curr = head
        while (curr and curr.next):
            if (curr.val == curr.next.val):
                curr.next = curr.next.next
            else:
                curr = curr.next
        
        return head
    
    @staticmethod
    def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        while (n > 0):
            if (nums1[m - 1] >= nums2[n - 1] and m > 0):
                nums1[m + n - 1] = nums1[m - 1]
                m -= 1
            else:
                nums1[m + n - 1] = nums2[n - 1]
                n -= 1

        return nums1
    
    @staticmethod
    def generate_pascal_triangle(numRows: int) -> List[List[int]]:

        def generate_row(row_list: List[int]) -> List[int]:
            result = [1]
            for i in range(1, len(row_list), 1):
                last_int = row_list[:i][-1]
                result.append(last_int + row_list[i])
            result.append(1)
            return result
        
        result = []

        for i in range(numRows):
            if (len(result)):
                last_row = result[-1]
                result.append(generate_row(last_row))
            else:
                result.append([1])

        return result
    
    @staticmethod
    def contains_duplicate(nums: List[int]) -> bool:
        seen_nums = set()

        for n in nums:
            if (n in seen_nums):
                return True
            else:
                seen_nums.add(n)

        return False
    
    @staticmethod
    def contains_nearby_duplicate(nums: List[int], k: int) -> bool:
        indices = {}

        for i, num in enumerate(nums):
            if (num in indices and abs(i - indices[num]) <= k):
                return True
            else:
                indices[num] = i

        return False
    
    @staticmethod
    def summary_ranges(nums: List[int]) -> List[str]:
        if (not nums):
            return []

        ranges = []

        start = 0
        for i in range(1, len(nums), 1):
            if (nums[i] != nums[i - 1] + 1):
                if (i == start + 1):
                    ranges.append(f'{nums[start]}')
                else:
                    ranges.append(f'{nums[start]}->{nums[i - 1]}')
                start = i

        if (start == len(nums) - 1):
            ranges.append(f'{nums[-1]}')
        else:
            ranges.append(f'{nums[start]}->{nums[-1]}')

        return ranges
    
    @staticmethod
    def summary_ranges_opt(nums: List[int]) -> List[str]: # use list comp
        ranges = []
        for n in nums:
            if ranges and ranges[-1][1] == n - 1:
                ranges[-1][1] = n
            else:
                ranges.append([n, n])

        return [f'{x}->{y}' if x != y else f'{x}' for x, y in ranges]
    
    @staticmethod
    def max_profit(prices: List[int]) -> int:
        max_profit = 0
        buy_price = prices[0]

        for price in prices[1:]:
            if (price > buy_price):
                max_profit = max(price - buy_price, max_profit)
            else:
                buy_price = price
        
        return max_profit

    @staticmethod
    def missing_number_opt(nums: List[int]) -> int:
        exp_nums_sum = len(nums) * (len(nums) + 1) // 2
        nums_sum = sum(nums)

        return exp_nums_sum - nums_sum
    
    @staticmethod
    def move_zeroes(nums: List[int]) -> None:

        if (not len(nums)):
            return

        i = 0
        while i < len(nums):

            if (nums[i] == 0):
                swap_i = i + 1
                
                while (swap_i < len(nums) and nums[swap_i] == 0):
                    swap_i += 1

                if (swap_i < len(nums)):
                    nums[i], nums[swap_i] = nums[swap_i], nums[i]

            i += 1

    @staticmethod
    def move_zeroes_opt(nums: List[int]) -> None:
        swap_i = 0
        for i in range(1, len(nums)):
            if (nums[i] != 0 and nums[swap_i] == 0):
                nums[i], nums[swap_i] = nums[swap_i], nums[i]

            if (nums[swap_i] != 0):
                swap_i += 1

    @staticmethod
    def has_word_pattern(pattern: str, s: str) -> bool:

        def create_pattern(s: Iterable[str] | str) -> List[str]:
            pattern = []
    
            keys = {}
            letter_ascii = ord('a')
            for c in s:
                if (c in keys):
                    letter = keys[c]
                    pattern.append(letter)
                else:
                    letter = chr(letter_ascii)
                    keys[c] = letter
                    pattern.append(letter)
                    letter_ascii += 1
            return pattern

        word_pattern = create_pattern(s.split())
        letter_pattern = create_pattern(pattern)

        return (word_pattern == letter_pattern)
    
    @staticmethod
    def max_intersect(nums1: List[int], nums2: List[int]) -> List[int]:
        result = []

        def count_nums(d: dict, nums: List[int]) -> dict:
            for num in nums:
                if (num in d):
                    d[num] += 1
                else:
                    d[num] = 1
            return d


        n1_counts = count_nums({}, nums1)
        n2_counts = count_nums({}, nums2)

        for num in n1_counts:
            if (num in n2_counts):
                for _ in range(min(n1_counts[num], n2_counts[num])):
                    result.append(num)

        return result
    
    @staticmethod
    def list_missing_numbers(nums: List[int]) -> List[int]:
        for num in nums:
            if (num < 0):
                num *= -1
            if (nums[num - 1] > 0):
                nums[num - 1] *= -1

        result = []
        for i, num in enumerate(nums):
            if (num > 0):
                result.append(i + 1)

        return result

    @staticmethod
    def count_matches_kids_by_min(g: List[int], s: List[int]) -> int:
        if (len(s) < 1):
            return 0

        g_counts = Counter(g)
        s_counts = Counter(s)

        match_count = 0
        biggest_need = max(g)
        avail_cookies = s
        while (biggest_need > 0 and len(avail_cookies) > 0):

            matching_cookie = min([ cookie for cookie in avail_cookies if cookie >= biggest_need ], default=0)
            if (matching_cookie > 0):
                match_count += 1
                if (g_counts[biggest_need] == 1):
                    del g_counts[biggest_need]
                else:
                    g_counts[biggest_need] -= 1
                if (s_counts[matching_cookie] == 1):
                    del s_counts[matching_cookie]
                else:
                    s_counts[matching_cookie] -= 1
            else:
                del g_counts[biggest_need]

            biggest_need = max(g_counts, default=0)
            avail_cookies = []
            for size, count in s_counts.items():
                for _ in range(count):
                    avail_cookies.append(size)

        return match_count
    
    @staticmethod
    def count_matches_kids_by_min_opt(g: List[int], s: List[int]) -> int:
        if (len(s) == 0):
            return 0

        g.sort()
        s.sort()

        match_count = 0
        gi = 0
        for cookie in s:
            
            if (cookie >= g[gi]):
                match_count += 1
                gi += 1
            
            if (gi == len(g)):
                return match_count
            
        return match_count
    
    @staticmethod
    def calc_island_perimeter(grid: List[List[int]]) -> int:
        perimeter = 0

        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if (cell == 1):
                    perimeter += 4
                    if (r - 1 >= 0 and grid[r - 1][c] == 1):
                        perimeter -= 2
                    if (c - 1 >= 0  and grid[r][c - 1] == 1):
                        perimeter -= 2

        return perimeter
    
    @staticmethod
    def calc_poisoned_duration(timeSeries: List[int], duration: int) -> int:
        if (len(timeSeries) < 2):
            return duration

        sum = 0

        for i, time in enumerate(timeSeries):
            if (i == len(timeSeries) - 1):
                sum += duration
            else:
                next_time = timeSeries[i + 1]
                if (next_time - time < duration):
                    sum += next_time - time
                else:
                    sum += duration

        return sum
    
    @staticmethod
    def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
        stack, num2_ngn_map = [], {}

        for num2 in nums2:
            while (stack and num2 > stack[-1]):
                num2_ngn_map[stack.pop()] = num2
            stack.append(num2)

        return [ num2_ngn_map.get(num1, -1) for num1 in nums1 ]
    
    @staticmethod
    def find_words_typable_one_kb_row(words: List[str]) -> List[str]:
        letter_map = {}
        for i, row in enumerate(["qwertyuiop", "asdfghjkl", "zxcvbnm"]):
            for letter in row:
                letter_map[letter] = i

        result = []

        for word in words:
            target_row = letter_map[word[0].lower()]
            add_it = True
            for letter in word:
                letter = letter.lower()
                if (letter_map[letter] != target_row):
                    add_it = False
                    break
            if (add_it):
                result.append(word)
            else:
                add_it = True

        return result
    
    @staticmethod
    def scores_to_medal_placements(scores: List[int]) -> List[str]:
        medals = [None, 'Gold Medal', 'Silver Medal', 'Bronze Medal']
        suffixes = [None, 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']

        def rank_to_place(rank: int) -> str:
            if (rank < 4):
                return medals[rank]
            else:
                return f'{rank}{suffixes[rank % 10]}'

        sorted_scores = sorted(scores, reverse=True)
        score_rank_map = {}
        for place, score in enumerate(sorted_scores):
            score_rank_map[score] = place + 1

        for i, score in enumerate(scores):
            scores[i] = rank_to_place(score_rank_map[score])

        return scores
    
    @staticmethod
    def reshape_matrix(matrix: List[List[int]], r: int, c: int) -> List[List[int]]:
        if (r * c != len(matrix) * len(matrix[0])):
            return matrix
        
        result = [[0] * c for _ in range(r)]
        cell_count = 0
        ri, ci = 0, 0
        for row in matrix:
            for cell in row:
                result[ri][ci] = cell
                cell_count += 1
                ri, ci = cell_count // c, cell_count % c

        return result
    
    @staticmethod
    def longest_harmonious_subsequence(nums: List[int]) -> int:
        if (len(nums) < 2):
            return 0
        
        num_counts = Counter(nums)
        diff_nums = list(num_counts.keys())
        diff_nums.sort()

        lhm = 0

        for i in range(len(diff_nums) - 1):
            i_num = diff_nums[i]
            j_num = diff_nums[i + 1]
            if (j_num == i_num + 1):
                lhm = max(lhm, num_counts[i_num] + num_counts[j_num])

        return lhm
    
    @staticmethod
    def area_of_max_int_post_op_incs(m: int, n: int, ops: List[List[int]]) -> int:
        if (not ops):
            return m * n
        
        mins = (ops[0][0], ops[0][1])
        for op in ops:
            mins = (min(mins[0], op[0]), min(mins[1], op[1]))

        return mins[0] * mins[1]


def vals_equal(list1: ListNode = None, list2: ListNode = None) -> bool:
    try:
        while (list1 and list2):
            if (list1.val != list2.val):
                return False
            else:
                list1 = list1.next
                list2 = list2.next
    except Exception:
        return False
    
    return True

@staticmethod
def nearest_common_restaurant_by_rank(list1: List[str], list2: List[str]) -> List[str]:
    pass


test_node1 = ListNode(1)
test_node2 = ListNode(2)
test_node3 = ListNode(4)
test_node1.next = test_node2
test_node2.next = test_node3
test_node_a = ListNode(1)
test_node_b = ListNode(3)
test_node_c = ListNode(4)
test_node_a.next = test_node_b
test_node_b.next = test_node_c
test_node_u = ListNode(1)
test_node_v = ListNode(1)
test_node_w = ListNode(2)
test_node_x = ListNode(3)
test_node_y = ListNode(4)
test_node_z = ListNode(4)
test_node_u.next = test_node_v
test_node_v.next = test_node_w
test_node_w.next = test_node_x
test_node_x.next = test_node_y
test_node_y.next = test_node_z
assert vals_equal(ListUtils.merge_two_lists(test_node1, test_node_a), test_node_u)
assert vals_equal(ListUtils.merge_two_lists(ListNode(), ListNode()), ListNode())
assert vals_equal(ListUtils.merge_two_lists(ListNode(), ListNode(0)), ListNode(0))

nums = [1, 1, 2]
expected_nums = [1, 2, 2]
assert ListUtils.remove_duplicates(nums) == 2
assert expected_nums == nums
nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
expected_nums = [0, 1, 2, 3, 4, 2, 2, 3, 3, 4]
assert ListUtils.remove_duplicates(nums) == 5
assert expected_nums == nums

nums = [3, 2, 2, 3]
expected_nums = [2, 2, 2, 3]
count = ListUtils.remove_element(nums, 3)
assert (count == 2)
assert (nums == expected_nums)
nums = [0, 1, 2, 2, 3, 0, 4, 2]
expected_nums =  [0, 1, 3, 0, 4, 0, 4, 2]
count = ListUtils.remove_element(nums, 2)
assert (count == 5)
assert (nums == expected_nums)

i = ListUtils.search_insert([1,3,5,6], 5)
assert (i == 2)
i = ListUtils.search_insert([1,3,5,6], 2)
assert (i == 1)
i = ListUtils.search_insert([1,3,5,6], 7)
assert (i == 4)

result = ListUtils.plus_one([1, 2, 3])
assert (result == [1, 2, 4])
result = ListUtils.plus_one([4, 3, 2, 1])
assert (result == [4, 3, 2, 2])
result = ListUtils.plus_one([9])
assert (result == [1, 0])

test_node1 = ListNode(1)
test_node2 = ListNode(1)
test_node3 = ListNode(2)
test_node1.next = test_node2
test_node2.next = test_node3
test_node_a = ListNode(1)
test_node_b = ListNode(2)
test_node_a.next = test_node_b
test_node_u = ListNode(1)
test_node_v = ListNode(1)
test_node_w = ListNode(2)
test_node_x = ListNode(3)
test_node_y = ListNode(3)
test_node_u.next = test_node_v
test_node_v.next = test_node_w
test_node_w.next = test_node_x
test_node_x.next = test_node_y
testNodeA = ListNode(1)
testNodeB = ListNode(2)
testNodeC = ListNode(3)
testNodeA.next = testNodeB
testNodeB.next = testNodeC
assert vals_equal(ListUtils.delete_duplicates(test_node1), test_node_a)
assert vals_equal(ListUtils.delete_duplicates(test_node_u), testNodeA)

result = ListUtils.merge([1, 2, 3, 0, 0, 0], 3, [2, 5, 6], 3)
assert (result == [1, 2, 2, 3, 5, 6])
result = ListUtils.merge([1], 1, [], 0)
assert (result == [1])
result = ListUtils.merge([0], 0, [1], 1)
assert (result == [1])

result = ListUtils.generate_pascal_triangle(5)
assert (result == [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]])
result = ListUtils.generate_pascal_triangle(1)
assert (result == [[1]])

result = ListUtils.contains_duplicate([1, 2, 3, 1])
assert (result == True)
result = ListUtils.contains_duplicate([1, 2, 3, 4])
assert (result == False)
result = ListUtils.contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2])
assert (result == True)

result = ListUtils.contains_nearby_duplicate([1, 2, 3, 1], 3)
assert (result == True)
result = ListUtils.contains_nearby_duplicate([1, 0, 1, 1], 1)
assert (result == True)
result = ListUtils.contains_nearby_duplicate([1, 2, 3, 1, 2, 3], 2)
assert (result == False)

result = ListUtils.summary_ranges([0, 1, 2, 4, 5, 7])
assert (result == ["0->2", "4->5", "7"])
result = ListUtils.summary_ranges([0, 2, 3, 4, 6, 8, 9])
assert (result == ["0", "2->4", "6", "8->9"])

result = ListUtils.summary_ranges_opt([0, 1, 2, 4, 5, 7])
assert (result == ["0->2", "4->5", "7"])
result = ListUtils.summary_ranges_opt([0, 2, 3, 4, 6, 8, 9])
assert (result == ["0", "2->4", "6", "8->9"])

result = ListUtils.max_profit([7, 1, 5, 3, 6, 4])
assert (result == 5)
result = ListUtils.max_profit([7, 6, 4, 3, 1])
assert (result == 0)

result = ListUtils.missing_number_opt([3, 0, 1])
assert (result == 2)
result = ListUtils.missing_number_opt([0, 1])
assert (result == 2)
result = ListUtils.missing_number_opt([9, 6, 4, 2, 3, 5, 7, 0, 1])
assert (result == 8)

nums = [0, 1, 0, 3, 12]
ListUtils.move_zeroes(nums)
assert (nums == [1, 3, 12, 0, 0])
nums = [0]
ListUtils.move_zeroes(nums)
assert (nums == [0])
nums = [0, 1, 2, 0, 0, 3, 4]
ListUtils.move_zeroes(nums)
assert (nums == [1, 2, 3, 4, 0, 0, 0])

nums = [0, 1, 0, 3, 12]
ListUtils.move_zeroes_opt(nums)
assert (nums == [1, 3, 12, 0, 0])
nums = [0]
ListUtils.move_zeroes_opt(nums)
assert (nums == [0])
nums = [0, 1, 2, 0, 0, 3, 4]
ListUtils.move_zeroes_opt(nums)
assert (nums == [1, 2, 3, 4, 0, 0, 0])

result = ListUtils.has_word_pattern("abba", "dog cat cat dog")
assert (result == True)
result = ListUtils.has_word_pattern("abba", "dog cat cat dog dog")
assert (result == False)
result = ListUtils.has_word_pattern("abba", "dog cat cat fish")
assert (result == False)
result = ListUtils.has_word_pattern("aaaa", "dog cat cat dog")
assert (result == False)

result = ListUtils.max_intersect([1, 2, 2, 1], [2, 2])
result.sort()
assert (result == [2, 2])
result = ListUtils.max_intersect([4, 9, 5], [9, 4, 9, 8, 4])
result.sort()
assert (result == [4, 9])

result = ListUtils.list_missing_numbers([4, 3, 2, 7, 8, 2, 3, 1])
assert (result == [5, 6])
result = ListUtils.list_missing_numbers([1, 1])
assert (result == [2])
result = ListUtils.list_missing_numbers([1, 1, 2, 2, 3, 3, 3, 3])
assert (result == [4, 5, 6, 7, 8])

result = ListUtils.count_matches_kids_by_min([1, 2, 3], [1, 1])
assert (result == 1)
result = ListUtils.count_matches_kids_by_min([1, 2], [1, 2, 3])
assert (result == 2)
result = ListUtils.count_matches_kids_by_min([1, 1, 1, 2, 3, 3, 4, 4], [1, 1, 2, 3, 3, 3, 4])
assert (result == 7)

result = ListUtils.count_matches_kids_by_min_opt([1, 2, 3], [1, 1])
assert (result == 1)
result = ListUtils.count_matches_kids_by_min_opt([1, 2], [1, 2, 3])
assert (result == 2)
result = ListUtils.count_matches_kids_by_min_opt([1, 1, 1, 2, 3, 3, 4, 4], [1, 1, 2, 3, 3, 3, 4])
assert (result == 7)

result = ListUtils.calc_island_perimeter([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 0, 0]])
assert (result == 16)
result = ListUtils.calc_island_perimeter([[1]])
assert (result == 4)
result = ListUtils.calc_island_perimeter([[1, 0]])
assert (result == 4)

result = ListUtils.calc_poisoned_duration([1, 4], 2)
assert (result == 4)
result = ListUtils.calc_poisoned_duration([1, 2], 2)
assert (result == 3)

result = ListUtils.next_greater_element([4, 1, 2], [1, 3, 4, 2])
assert (result == [-1, 3, -1])
result = ListUtils.next_greater_element([2, 4], [1, 2, 3, 4])
assert (result == [3, -1])

result = ListUtils.find_words_typable_one_kb_row(["Hello", "Alaska", "Dad", "Peace"])
assert (result == ["Alaska", "Dad"])
result = ListUtils.find_words_typable_one_kb_row(["omk"])
assert (result == [])
result = ListUtils.find_words_typable_one_kb_row(["adsdf","sfd"])
assert (result == ["adsdf","sfd"])

result = ListUtils.scores_to_medal_placements([5, 4, 3, 2, 1])
assert (result == ["Gold Medal", "Silver Medal", "Bronze Medal", "4th", "5th"])
result = ListUtils.scores_to_medal_placements([10, 3, 8, 9, 4])
assert (result == ["Gold Medal", "5th", "Bronze Medal", "Silver Medal", "4th"])

result = ListUtils.reshape_matrix([[1, 2], [3, 4]], 1, 4)
assert (result == [[1, 2, 3, 4]])
result = ListUtils.reshape_matrix([[1, 2], [3, 4]], 2, 4)
assert (result == [[1, 2], [3, 4]])
result = ListUtils.reshape_matrix([[1, 2, 3, 4]], 2, 2)
assert (result == [[1, 2], [3, 4]])

result = ListUtils.longest_harmonious_subsequence([1, 3, 2, 2, 5, 2, 3, 7])
assert (result == 5)
result = ListUtils.longest_harmonious_subsequence([1, 2, 3, 4])
assert (result == 2)
result = ListUtils.longest_harmonious_subsequence([1, 1, 1, 1])
assert (result == 0)

result = ListUtils.area_of_max_int_post_op_incs(3, 3, [[2, 2],[3, 3]])
assert (result == 4)
result = ListUtils.area_of_max_int_post_op_incs(3, 3, [[2, 2],[3, 3],[3, 3],[3, 3],[2, 2],[3, 3],[3, 3],[3, 3],[2, 2],[3, 3],[3, 3],[3, 3]])
assert (result == 4)
result = ListUtils.area_of_max_int_post_op_incs(3, 3, [])
assert (result == 9)
