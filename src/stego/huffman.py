from collections import deque
from typing import List, Optional


class Node:
    # node of tree
    def __init__(self,
                 prob: float,
                 left=None,
                 right=None,
                 index: Optional[int] = None,
                 search_path: Optional[int] = None) -> None:
        # `search_path is None` - no need
        # `search_path == 0` - self
        # `search_path == -1` - left subtree
        # `search_path == 1` - right subtree

        self.left = left
        self.right = right
        self.prob = prob
        self.index = index
        self.search_path = search_path

    def __lt__(self, c) -> bool:
        return self.prob < c.prob

    def is_leaf(self) -> bool:
        return self.index is not None


# def create_huffman_tree(indices: List[int], probs: List[float], search_for: Optional[int] = None) -> Node:
#     import heapq as hq
#     sz = len(indices)
#     nodes = [Node(probs[i], index=indices[i], search_path=(0 if search_for == indices[i] else None)) for i in range(sz)]
#     hq.heapify(nodes)
#     while len(nodes) > 1:
#         left = hq.heappop(nodes)
#         right = hq.heappop(nodes)
#         prob = left.prob + right.prob
#         search_path = None
#         if left.search_path is not None:
#             search_path = -1
#         elif right.search_path is not None:
#             search_path = 1
#         hq.heappush(nodes, Node(prob, left, right, search_path=search_path))
#     root = nodes[0]
#     return root


def create_huffman_tree(indices: List[int], probs: List[float], search_for: Optional[int] = None) -> Node:

    sz = len(indices)
    nodes = [Node(probs[i], index=indices[i], search_path=(0 if search_for == indices[i] else None)) for i in range(sz)]
    # nodes.sort()  # maybe already sorted
    # nodes = nodes.reverse()
    a = deque(nodes)
    b = deque()

    def fun():
        nonlocal a, b
        if len(a) > 0 and len(b) > 0 and a[-1] < b[-1]:
            item = a.pop()
        elif len(a) == 0:
            item = b.pop()
        elif len(b) == 0:
            item = a.pop()
        else:
            item = b.pop()
        return item

    while len(a) + len(b) > 1:
        left = fun()
        right = fun()
        prob = left.prob + right.prob
        search_path = None
        if left.search_path is not None:
            search_path = -1
        elif right.search_path is not None:
            search_path = 1
        b.appendleft(Node(prob, left, right, search_path=search_path))
    root = b[0] if len(b) > 0 else a[0]
    return root


if __name__ == '__main__':
    indices = [0, 1, 2, 3, 4, 5, 6, 7]
    probs = [0.2, 0.145, 0.136, 0.125, 0.125, 0.114, 0.105, 0.05]
    tree = create_huffman_tree(indices, probs)
    print('end')
