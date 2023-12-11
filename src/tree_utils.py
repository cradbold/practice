from typing import Optional, Callable, List, Any
from collections import Counter

class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class  NAryNode:

    def __init__(self, val=0, children=[]):
        self.val = val
        self.children = children

class TreeUtils:

    @staticmethod
    def in_order_traversal(root: Optional[TreeNode]) -> List[int]:
        subtree = []
        result = []

        while (root or subtree):
            while root:
                subtree.append(root)
                root = root.left
            
            root = subtree.pop()
            result.append(root.val)
            root = root.right
        
        return result
    
    @staticmethod
    def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if (not p and not q):
            return True
        
        if (not p or not q or (p.val != q.val)):
            return False

        return TreeUtils.is_same_tree(p.left, q.left) and TreeUtils.is_same_tree(p.right, q.right)
    
    @staticmethod
    def is_symmetric(root: Optional[TreeNode]) -> bool:
        
        def solve(left, right):
            if ((left and not right) or (not left and right)):
                return False
            elif (left == right):
                return True
            elif (left.val != right.val):
                return False
            else:
                return solve(left.left, right.right) and solve(left.right, right.left)
            
        return solve(root.left, root.right)
    
    @staticmethod
    def max_depth(root: Optional[TreeNode]) -> int:
        
        if (root):
            return (1 + max(TreeUtils.max_depth(root.left), TreeUtils.max_depth(root.right)))
        else:
            return 0
        
    @staticmethod
    def sorted_array_to_bst(nums: List[int]) -> Optional[TreeNode]:
        total_nums = len(nums)
        if (not total_nums):
            return None
        
        mid_point = total_nums // 2
        return TreeNode(nums[mid_point], TreeUtils.sorted_array_to_bst(nums[:mid_point]), TreeUtils.sorted_array_to_bst(nums[mid_point + 1:]))

    @staticmethod
    def sum_left_leaves_iter(root: Optional[TreeNode]) -> int:
        left_sum = 0
        subtrees = []

        while (root or subtrees):
            while (root):
                subtrees.append(root)
                if (root.left and not root.left.left and not root.left.right):
                    left_sum += root.left.val
                root = root.left
            
            root = subtrees.pop()
            root = root.right

        return left_sum
    
    @staticmethod
    def sum_left_leaves_rec(root: Optional[TreeNode]) -> int:
        if (not root):
            return 0

        if (root.left and not root.left.left and not root.left.right):
            return root.left.val + TreeUtils.sum_left_leaves_rec(root.right)
        else:
            return TreeUtils.sum_left_leaves_rec(root.left) + TreeUtils.sum_left_leaves_rec(root.right)

    @staticmethod
    def find_mode(root: Optional[TreeNode]) -> List[int]:
        num_counts = Counter()
        root_modes, subtrees = [], []
        highest_count = 1

        while (root or subtrees):
            while (root):
                subtrees.append(root)
                root = root.left
            
            root = subtrees.pop()
            num = root.val

            num_counts[num] += 1
            highest_count_candidate = num_counts[num]

            if (highest_count_candidate == highest_count):
                root_modes.append(num)
            elif (highest_count_candidate > highest_count):
                highest_count = highest_count_candidate
                root_modes = [num]
            
            root = root.right

        return root_modes
    
    @staticmethod
    def min_node_diff(root: Optional[TreeNode]) -> int:
        min_diff = abs(root.val - root.left.val) if (root.left) else abs(root.val - root.right.val)
        prev_val = float('-inf')
        stack = []

        while (root or stack):
            while (root):
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            min_diff = min(min_diff, root.val - prev_val)
            prev_val = root.val
            root = root.right

        return min_diff
    
    @staticmethod
    def diameter_iter(root: Optional[TreeNode]) -> int:
        diameter = 0
        depths = {}
        stack = [ (root, False) ]

        while (stack):
            (node, process_now) = stack.pop()
            if (node):
                if (process_now):
                    left_depth = depths.pop(node.left) if (node.left) else 0
                    right_depth = depths.pop(node.right) if (node.right) else 0
                    diameter = max(diameter, left_depth + right_depth)
                    depths[node] = 1 + max(left_depth, right_depth)
                else:
                    stack.append((node, True))
                    stack.append((node.left, False))
                    stack.append((node.right, False))

        return diameter

    @staticmethod
    def diameter_rec(root: Optional[TreeNode]) -> int:
        
        longest_diameter = 0

        def depth(node: Optional[TreeNode]) -> int:
            nonlocal longest_diameter
            left_depth = depth(node.left) if (node.left) else 0
            right_depth = depth(node.right) if (node.right) else 0
            longest_diameter = max(longest_diameter, left_depth + right_depth)
            return 1 + max(left_depth, right_depth)

        depth(root)
        return longest_diameter
    
    @staticmethod
    def n_ary_depth_iter(root: Optional[NAryNode]) -> int:
        if (not root):
            return 0

        child_depth = 0
        depths = {}

        stack = [(root, False)]
        while (stack):
            (node, process_now) = stack.pop()
            if (node):
                if (process_now):
                    max_child_depth = 0
                    for child in node.children:
                        max_child_depth = max(max_child_depth, depths[child])
                    child_depth = max(child_depth, max_child_depth)
                    depths[node] = 1 + max_child_depth
                else:
                    stack.append((node, True))
                    for child in node.children:
                        stack.append((child, False))

        return 1 + child_depth
    
    @staticmethod
    def n_ary_depth_rec(root: Optional[NAryNode]) -> int:
        pass


tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(3)
tn1.right = tn2
tn2.left = tn3
result = TreeUtils.in_order_traversal(tn1)
assert (result == [1, 3, 2])
result = TreeUtils.in_order_traversal(None)
assert (result == [])
result = TreeUtils.in_order_traversal(TreeNode(1))
assert (result == [1])

tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(3)
tn1.left = tn2
tn1.right = tn3
tnA = TreeNode(1)
tnB = TreeNode(2)
tnC = TreeNode(3)
tnA.left = tnB
tnA.right = tnC
result = TreeUtils.is_same_tree(tn1, tnA)
assert (result == True)
tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn1.left = tn2
tnA = TreeNode(1)
tnB = TreeNode(2)
tnA.right = tnB
result = TreeUtils.is_same_tree(tn1, tnA)
assert (result == False)
tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(1)
tn1.left = tn2
tn1.right = tn3
tnA = TreeNode(1)
tnB = TreeNode(1)
tnC = TreeNode(2)
tnA.left = tnB
tnA.right = tnC
result = TreeUtils.is_same_tree(tn1, tnA)
assert (result == False)

tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(2)
tn4 = TreeNode(3)
tn5 = TreeNode(4)
tn6 = TreeNode(4)
tn7 = TreeNode(3)
tn1.left = tn2
tn1.right = tn3
tn2.left = tn4
tn2.right = tn5
tn3.left = tn6
tn3.right = tn7
result = TreeUtils.is_symmetric(tn1)
assert (result == True)
tn3.right = None
result  = TreeUtils.is_symmetric(tn1)
assert (result == False)
tn2.left = None
result = TreeUtils.is_symmetric(tn1)
assert (result == True)
tn2.left = TreeNode(2)
tn2.right = None
tn3.left = TreeNode(2)
tn3.right = None
result = TreeUtils.is_symmetric(tn1)
assert (result == False)

tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(3)
tn4 = TreeNode(4)
tn5 = TreeNode(5)
tn1.left = tn2
tn1.right = tn3
tn3.left = tn4
tn3.right = tn5
result = TreeUtils.max_depth(tn1)
assert (result == 3)
tn1.left = None
tn3.left = None
tn3.right = None
result = TreeUtils.max_depth(tn1)
assert (result == 2)
tn1.left = tn2
result = TreeUtils.max_depth(tn1)
assert (result == 2)

result = TreeUtils.sorted_array_to_bst([-10, -3, 0, 5, 9])
# print(result)
# assert (result == 3)
result = TreeUtils.sorted_array_to_bst([1, 3])
# print(result)
# assert (result == 3)

tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(3)
tn4 = TreeNode(4)
tn5 = TreeNode(5)
tn1.left = tn2
tn1.right = tn3
tn3.left = tn4
tn3.right = tn5
result = TreeUtils.sum_left_leaves_iter(tn1)
assert (result == 6)
result = TreeUtils.sum_left_leaves_rec(tn1)
assert (result == 6)
tn1.left = tn5
tn3.right = None
result = TreeUtils.sum_left_leaves_iter(tn1)
assert (result == 9)
result = TreeUtils.sum_left_leaves_rec(tn1)
assert (result == 9)
tn5.right = tn2
result = TreeUtils.sum_left_leaves_iter(tn1)
assert (result == 4)
result = TreeUtils.sum_left_leaves_rec(tn1)
assert (result == 4)

tn1 = TreeNode(4)
tn2 = TreeNode(2)
tn3 = TreeNode(6)
tn4 = TreeNode(1)
tn5 = TreeNode(3)
tn1.left = tn2
tn1.right = tn3
tn2.left = tn4
tn2.right = tn5
result = TreeUtils.min_node_diff(tn1)
assert (result == 1)
tn1 = TreeNode(1)
tn2 = TreeNode(0)
tn3 = TreeNode(48)
tn4 = TreeNode(12)
tn5 = TreeNode(49)
tn1.left = tn2
tn1.right = tn3
tn3.left = tn4
tn3.right = tn5
result = TreeUtils.min_node_diff(tn1)
assert (result == 1)

tn1 = TreeNode(1)
tn2 = TreeNode(2)
tn3 = TreeNode(3)
tn4 = TreeNode(4)
tn5 = TreeNode(5)
tn1.left = tn2
tn1.right = tn3
tn2.left = tn4
tn2.right = tn5
result = TreeUtils.diameter_iter(tn1)
assert (result == 3)
result = TreeUtils.diameter_rec(tn1)
assert (result == 3)
tn1.right = None
tn2.left = None
tn2.right = None
result = TreeUtils.diameter_iter(tn1)
assert (result == 1)
result = TreeUtils.diameter_rec(tn1)
assert (result == 1)

nan_root = NAryNode(1, [NAryNode(3, [NAryNode(5, []), NAryNode(6, [])]), NAryNode(2, []), NAryNode(4, [])])
result = TreeUtils.n_ary_depth_iter(nan_root)
assert (result == 3)
result = TreeUtils.n_ary_depth_rec(nan_root)
assert (result == 3)
nan_root = NAryNode(1, [
    NAryNode(2, []), 
    NAryNode(3, [NAryNode(6, []), NAryNode(7, [NAryNode(11, [NAryNode(14, [])])])]),
    NAryNode(4, [NAryNode(8, [NAryNode(12, [])])]), 
    NAryNode(5, [NAryNode(9, [NAryNode(13, [])]), NAryNode(10, [])])
])
result = TreeUtils.n_ary_depth_iter(nan_root)
assert (result == 5)
result = TreeUtils.n_ary_depth_rec(nan_root)
assert (result == 5)
