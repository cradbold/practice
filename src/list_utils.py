from typing import Optional, Callable, List, Any

class ListNode:

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class ListUtils:

    @staticmethod
    def merge_two_lists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = head = ListNode()
        
        while list1 and list2:               
            if list1.val < list2.val:
                cur.next = list1
                list1, cur = list1.next, list1
            else:
                cur.next = list2
                list2, cur = list2.next, list2

        if (list1):
            cur.next = list1

        if (list2):
            cur.next = list2
            
        return head.next


def assert_list_utils(func: Callable, args: List, val: Any) -> None:
    print(f'Calling {func.__name__} with args: {args} and asserting return value: {val}')
    result = func(*args)
    print(f'  Result: {result == val}')
    assert result == val

assert_list_utils(ListUtils.merge_two_lists, [[1, 2, 4], [1, 3, 4]], [1, 1, 2, 3, 4, 4])
assert_list_utils(ListUtils.merge_two_lists, [[], []], [])
assert_list_utils(ListUtils.merge_two_lists, [[], [0]], [0])
