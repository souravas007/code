from collections import deque
from typing import Optional, List

from neetcode.trees.tree_node import TreeNode


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        queue = deque()
        result = []
        if root is None:
            return result
        queue.append(root)

        while len(queue) > 0:
            current_result = []
            size = len(queue)
            for i in range(size):
                element = queue.popleft()
                current_result.append(element.val)
                if element.left:
                    queue.append(element.left)
                if element.right:
                    queue.append(element.right)
            result.append(current_result.copy())
        return result
