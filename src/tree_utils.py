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
