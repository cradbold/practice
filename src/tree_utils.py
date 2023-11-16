from typing import Optional, Callable, List, Any

class TreeNode:

    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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
