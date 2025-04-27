import collections
from collections import defaultdict
import rust_encoder


class Node:
    def __init__(self, parent, number):
        """
        :param parent:
        :param number: Index in der sortierten color_sizes liste
        """
        self.number = number
        self.parent = parent
        self.children = []

    def is_leaf(self):
        return not self.children

    def __repr__(self):
        if not self.parent:
            return ""
        return f"{self.parent.__repr__()} {self.number}"

    def visualize(self, tree):
        tree.create_node(self.id(), self.id(), parent=self.parent.id() if self.parent else None)
        for c in self.children:
            c.visualize(tree)

    def get_depth(self, color_sizes):
        if not self.parent:
            return 0
        return self.parent.get_depth(color_sizes) + color_sizes[self.number]

    def reduce(self):
        """
        Removes nodes with a single leaf
        :return:
        """
        if len(self.children) == 1:
            child = self.children[0]
            child.number = self.number
            child.parent = self.parent
            if self.parent:
                self.parent.children.remove(self)
                self.parent.children.append(child)
            for c in child.children:
                c.reduce()
            return child
        else:
            for c in self.children:
                c.reduce()
            return self

    def copy(self):
        new_node = Node(None, self.number)
        new_children = []
        for c in self.children:
            c_copy = c.copy()
            c_copy.parent = new_node      # <— fix up the parent
            new_children.append(c_copy)
        new_node.children = new_children
        return new_node



    def delete(self):
        """
        Deletes the node and all its children
        :return:
        """
        if self.parent:
            self.parent.children.remove(self)
        self.parent = None

    def fill_node(self, r):
        """
        Attach as many leaves as possible to this node
        """
        for i in range(r):
            self.children.append(Node(self, i))
        return self

    def fill_up_to_height(self, h, color_sizes):
        """
        Attach so many leaves, that depth(l) >= h for every Leaf l
        :param h:
        :return:
        """
        if self.is_leaf():
            if self.get_depth(color_sizes) < h:
                self.fill_node(len(color_sizes))
            else:
                return
        for c in self.children:
            c.fill_up_to_height(h, color_sizes)


def generate_tree(Qs, color_sizes):
    color_sizes = sorted(color_sizes)
    root = Node(None, 0)
    Levels = defaultdict(list)
    Levels[0].append(root)
    for i, c in enumerate(color_sizes):
        root.children.append(Node(root, i))
        Levels[c].append(root.children[-1])

    for i, q in enumerate(Qs):
        if q == 0:
            continue
        for j, n in enumerate(Levels[i + 1]):
            for ii, c in enumerate(color_sizes):
                n.children.append(Node(n, ii))
                Levels[c + i + 1].append(n.children[-1])
            if j + 1 == q:
                break
    return root


def get_leaves_rec(node):
    if not node.children:
        return [node]
    else:
        a = []
        for c in node.children:
            a.extend(get_leaves_rec(c))
        return a

def get_nodes_rec(node):
    """
    Returns all nodes that are children of this node
    :param node:
    :return:
    """
    if not node.children:
        return [node]
    else:
        a = [node]
        for c in node.children:
            a.extend(get_nodes_rec(c))
        return a

def generate_code_from_tree(root, color_sizes):
    """
    :param root:
    :return: [(code, cost), (code, cost), ...]
    """
    leaves = get_leaves_rec(root)
    # get codes
    codes = []
    for l in leaves:
        code = []
        while l.parent:
            code.append(l.number)
            l = l.parent
        cost = 0
        for c in code:
            cost += color_sizes[c]


        codes.append((code[::-1], cost))
    return  codes

def is_prefixfree(codes):
    """
    Check if the codes are prefix free
    :param codes:
    :return:
    """
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if codes[i][0] == codes[j][0][:len(codes[i][0])]:
                return False
    return True

def attach_tree_at_leaves(main_root, sub_root, fixed_slots, color_sizes):
    """
    Attach the sub_tree at the leaves of the main_tree
    :param main_root:
    :param sub_root:
    :return:
    """
    leaves = get_leaves_rec(main_root)
    leaves = sorted(leaves, key=lambda x: x.get_depth(color_sizes))
    for i, leaf in enumerate(leaves[fixed_slots:]):
        s_r = sub_root.copy()
        leaf.parent.children.remove(leaf)
        leaf.parent.children.append(s_r)
        s_r.parent = leaf.parent
        s_r.number = leaf.number
    return main_root



def reduce_tree_to_height(root, h):
    """
    :param root:
    :param h:
    :return:
    """
    root = root.reduce()
    nodes = get_nodes_rec(root)
    for n in nodes:
        if n.get_depth(color_sizes) > h:
            n.delete()

    return root.reduce()



def calc_cost(text, codes):
    occurrences = [text.count(i) for i in set(text)]
    occurrences = sorted(occurrences, reverse=True)
    costs = sorted(i[1] for i in codes)
    total_cost = 0
    for i in range(len(occurrences)):
        total_cost += occurrences[i] * costs[i]
    return total_cost


def reduce(sig, n):
    new_sig = [min(sig[0], n)]
    for i in range(1, len(sig)):
        new_sig.append(min(sig[i], n - sum(new_sig)))
    return new_sig


def extend(sig, q, D):
    new_sig = [sig[0] + sig[1]] + sig[2:] + [0]
    x = ([-1] + D)
    for i in range(len(new_sig)):
        new_sig[i] = new_sig[i] + q * x[i]
    return new_sig



class Encoder:
    def __init__(self, frequencies, color_sizes):
        self.frequencies = frequencies
        self.num_colors = len(color_sizes)
        self.color_sizes = sorted(color_sizes)

    def encode_naive(self):
        return generate_code_from_tree(self.get_tree_naive(self.frequencies.values()), self.color_sizes)

    def get_tree_naive(self, frequencies): # TODO das ist nd bfs
        """
        Naive Methode: alle vollen Binärbäume werden mit Binärsuche durchsucht

        TODO Asymptotische Laufzeit von dem Ding hiers

        Beweis, dass es optimal ist
        :return:
        """
        # Probability for each character of the alphabet
        p = sorted(frequencies, reverse=True)
        # The number of different codewords needed to encode the text
        n = len(p)
        # The biggest Perl
        C = max(self.color_sizes)
        # Occurrences of each color size
        D = [self.color_sizes.count(i) for i in range(1, C + 1)]
        # number of pearls
        r = len(self.color_sizes)

        print(f"N: {n}, C : {C}")
        stack = collections.deque([([0] + D, 0, [])])  # (sig, cost, Qs(Expansions))
        best_cost = float("inf")
        best_tree = None
        num_visited = 0
        while stack:
            num_visited += 1
            sig, cost, Qs = stack.pop()
            new_cost = cost + sum([p[i] for i in range(sig[0], n)])
            if new_cost > best_cost:
                continue
            if sum(sig) > n * (r - 1):
                continue
            if sig[0] >= n:
                if cost < best_cost:
                    best_cost = cost
                    best_tree = Qs
                    #print(best_cost.__round__(4), len(stack))
                continue

            for q in range(0, sig[1] + 1):
                new_sig = extend(sig, q, D)
                stack.append((new_sig, new_cost, Qs + [q]))
        print("Num visited:", num_visited)
        return generate_tree(best_tree,self.color_sizes)

    def encode_reduce(self):
        return generate_code_from_tree(self.get_tree_reduce(self.frequencies.values()), self.color_sizes)

    def get_tree_reduce(self, frequencies):
        # Probability for each character of the alphabet
        p = sorted(frequencies, reverse=True)
        # The number of different codewords needed to encode the text
        n = len(p)
        # The biggest Perl
        C = max(self.color_sizes)
        # Occurrences of each color size
        D = [self.color_sizes.count(i) for i in range(1, C + 1)]
        # number of pearls
        r = len(self.color_sizes)

        print(f"Encode Reduce: N: {n}, C : {C}")
        stack = collections.deque([([0] + D, 0, [])])  # (sig, cost, Qs(Expansions))
        best_cost = float("inf")
        best_tree = None
        num_visited = 0
        while stack:
            num_visited += 1
            sig, cost, Qs = stack.pop()
            new_cost = cost + sum([p[i] for i in range(sig[0], n)])
            if new_cost > best_cost:
                continue

            if sig[0] >= n:
                if cost < best_cost:
                    best_cost = cost
                    best_tree = Qs
                    #print(best_cost.__round__(4), len(stack))
                continue

            for q in range(0, min(sig[1], n - sum(sig[i] for i in range(self.color_sizes[1] + 1))) + 1):
            #for q in range(0, min(sig[1], (n - sig[0] - sig[1]) // max(1, D[1])) + 1):
                new_sig = reduce(extend(sig, q, D), n)

                stack.append((new_sig, new_cost, Qs + [q]))
        #print("Num visited:", num_visited)
        return generate_tree(best_tree,self.color_sizes)

    def get_tree_rust(self, frequencies):
        qs = rust_encoder.encode_optimal_py(frequencies, self.color_sizes)
        return generate_tree(qs, color_sizes)

    def encode_rust(self):
        return generate_code_from_tree(self.get_tree_rust(list(self.frequencies.values())), self.color_sizes)

    def encode_big_optimize(self, text):
        n = len(self.frequencies)
        max_leaves = 40
        min_chunk_size = n // max_leaves + 1
        best_cost = float("inf")
        best_code = None
        for i in range(min_chunk_size, max_leaves):
            codes = self.encode_big(i)
            cost = calc_cost(text, codes)
            if cost < best_cost:
                best_cost = cost
                best_code = codes
                print("Best cost:", best_cost)
        return best_code


    def encode_big(self, chunk_size=50):
        cutoff = 0.05
        common = [i for i in self.frequencies.values() if i > cutoff]
        #print(f"N: {len(self.frequencies)}, C : {max(self.color_sizes)}")
        #print(f"Common: {len(common)}, Not Common: {len(self.frequencies) - len(common)}")
        not_common = sorted(i for i in self.frequencies.values() if i <= cutoff)

        # split not common in chunks of chunk_size
        chunks = []
        for i in range((len(not_common) // chunk_size)):
            chunks.append(not_common[i * chunk_size: (i + 1) * chunk_size])
        chunks.append(not_common[(len(not_common) // chunk_size) * chunk_size:])

        chunk_root = self.get_tree_rust(chunks[0])

        main_root = self.get_tree_rust(common + [sum(c) for c in chunks])
        main_root = attach_tree_at_leaves(main_root, chunk_root, len(common), color_sizes)
        return generate_code_from_tree(main_root, self.color_sizes)



def attach_tree_at_leaf(leaf, root):
    """
    Replace the leaf with root
    :param leaf:
    :param root:
    :return:
    """
    leaf.parent.children.remove(leaf)
    leaf.parent.children.append(root)
    root.parent = leaf.parent


def get_frequencies(text):
    return {i: text.count(i) / len(text) for i in set(text)}


def generate_full_tree_to_height(color_sizes, h):
    root = Node(None, None)
    root.fill_up_to_height(h, color_sizes)
    reduce_tree_to_height(root, h)
    return root

def assume_equal_probabilites(color_sizes, frequencies):
    """
    :param color_sizes:
    :param frequencies:
    :return:
    """
    for i in range(1, len(frequencies)):
        root = generate_full_tree_to_height(color_sizes, i)
        if len(get_leaves_rec(root)) >= len(frequencies):
            return generate_code_from_tree(root)

def parse_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        color_sizes = lines[1]
        text = lines[2][:-1] # remove \n
        color_sizes = list(map(int, color_sizes.split()))
    return color_sizes, text

if __name__ == '__main__':

    filename = ("Examples/schmuck9.txt")
    color_sizes, text = parse_file(filename)

    encoder = Encoder(get_frequencies(text), color_sizes)
    #code = encoder.encode_big_optimize(text)
    #code = encoder.encode_reduce() # 134563
    code = encoder.encode_rust()

    #code = generate_code_from_tree(generate_tree([2, 6, 15, 33, 31, 0, 0], color_sizes), color_sizes)
    print("Präfixfrei: ", is_prefixfree(code))
    print(len(code), code)
    print(calc_cost(text, code))
    exit()
