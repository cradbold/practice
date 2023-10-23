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
assert_list_utils(ListUtils.merge_two_lists, [test_node1, test_node_a], [1, 1, 2, 3, 4, 4])
assert_list_utils(ListUtils.merge_two_lists, [[], []], [])
assert_list_utils(ListUtils.merge_two_lists, [[], [ListNode(0)]], [0])
