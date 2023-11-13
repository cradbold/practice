from typing import Optional, Callable, List, Any

class ListNode:

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


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
        for i in range(len(digits)-1, -1, -1):
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
    def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> List[int]:
        a, b = m - 1, n - 1
        write_index = m + n - 1

        while (b >= 0):
            if (a >= 0 and nums1[a] > nums2[b]):
                nums1[write_index] = nums1[a]
                a -= 1
            else:
                nums1[write_index] = nums2[b]
                b -= 1

            write_index -= 1
                
        return nums1


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
