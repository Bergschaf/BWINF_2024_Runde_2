import heapq


class Node:
    def __init__(self, p, a=None):
        """
        :param p: Frequenz
        :param a: Buchstabe (none wenn interner Knoten)
        """
        self.p = p
        self.a = a
        self.children = []

    def __lt__(self, other):
        return self.p < other.p


def generate_code_from_tree(root):
    codes = {}
    stack = [(root, [])]  # (node, turns to get there)
    while stack:
        node, code = stack.pop()
        for i, c in enumerate(node.children):
            if c.a is not None:
                codes[c.a] = code + [i]
            else:
                stack.append((c, code + [i]))
    return codes


def encode_huffman(r, frequencies):
    """
    :param r: Maximale Kinder pro Knoten
    :param frequencies: Dictionary mit Buchstabe : Frequenz
    :return:
    """
    minHeap = []
    for a, p in frequencies.items():
        heapq.heappush(minHeap, Node(p, a))
    n = len(frequencies)
    m = 0
    while (n + m) % (r - 1) != 1: m += 1

    for _ in range(m):
        heapq.heappush(minHeap, Node(0))

    while len(minHeap) > 1:
        new_children = []
        for _ in range(r):
            new_children.append(heapq.heappop(minHeap))
        new_node = Node(sum(n.p for n in new_children))
        new_node.children = new_children
        heapq.heappush(minHeap, new_node)

    return generate_code_from_tree(minHeap[0])


def parse_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        color_sizes = lines[1]
        text = lines[2][:-1]  # remove \n
        color_sizes = list(map(int, color_sizes.split()))
    return color_sizes, text


def get_frequencies(text):
    return {i: text.count(i) / len(text) for i in text}


def calc_cost(text, code):
    return sum(len(code[i]) * text.count(i) for i in set(text))


def print_code(code, text):
    for letter, c in sorted(code.items(), key=lambda x: text.count(x[0]) - 0.1 * len(x[1]), reverse=True):
        print(f"{letter}({text.count(letter)}):\t {c}")
    print(f"Gesamtlänge der Perlenkette: {calc_cost(text, code)}")


if __name__ == '__main__':
    filename = "../Examples/schmuck00.txt"

    color_sizes, text = parse_file(filename)

    code = encode_huffman(len(color_sizes), get_frequencies(text))
    print_code(code, text)
