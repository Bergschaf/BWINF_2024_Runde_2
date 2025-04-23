from collections import defaultdict
from treelib import Node, Tree
import graphviz


def generate_tree(color_sizes, Qs):
    color_sizes = sorted(color_sizes)
    root = Node(0, None)
    Levels = defaultdict(list)
    Levels[0].append(root)
    for i, c in enumerate(color_sizes):
        root.children.append(Node(c, root, i))
        Levels[c].append(root.children[-1])

    for i, q in enumerate(Qs):
        if q == 0:
            continue
        j = 0
        for n in Levels[i + 1]:
            for ii, c in enumerate(color_sizes):
                n.children.append(Node(c + i + 1, n, ii))
                Levels[c + i + 1].append(n.children[-1])
            j += 1
            if j == q:
                break
    # count leaves
    leaves = []
    for i in Levels:
        for n in Levels[i]:
            if not n.children:
                leaves.append(n)
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

        codes.append((code,  cost))
    return root, codes

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


def all_sigs_rec(C, n):
    if C == 1:
        return [[i] for i in range(n + 1)]
    else:
        return [[i] + sig for i in range(n + 1) for sig in all_sigs_rec(C - 1, n - i)]


def all_sigs(C, n):
    return [sig for sig in all_sigs_rec(C, n) if sum(sig[1:]) != 0]


def lt_sig(sig1, sig2):
    sig1 = [sum(sig1[:i + 1]) for i in range(len(sig1))]
    sig2 = [sum(sig2[:i + 1]) for i in range(len(sig2))]
    for i in range(len(sig1)):
        if sig1[i] < sig2[i]:
            return True
        if sig1[i] > sig2[i]:
            return False
    return False


class Encoder:
    def __init__(self, frequencies, color_sizes):
        self.frequencies = frequencies
        self.num_colors = len(color_sizes)
        self.color_sizes = sorted(color_sizes)

    def encode_bfs(self):
        """
        Naive Methode: alle vollen Binärbäume werden mit Binärsuche durchsucht
        :return:
        """
        # Probability for each character of the alphabet
        p = sorted(self.frequencies.values(), reverse=True)
        # The number of different codewords needed to encode the text
        n = len(p)
        # The biggest Perl
        C = max(self.color_sizes)
        # Occurrences of each color size
        D = [self.color_sizes.count(i) for i in range(1, C + 1)]

        print(f"N: {n}, C : {C}")
        stack = [([0] + D, 0, [])] # (sig, cost, Qs(Expansions))
        best_cost = float("inf")
        possible_res = [] # (sig, cost, Qs#)
        while stack:
            sig, cost, Qs = stack.pop(0)
            new_cost = cost + sum([p[i] for i in range(sig[0], n)])
            if new_cost > best_cost:
                continue
            if sig[0] >= n:
                if cost < best_cost:
                    best_cost = cost
                possible_res.append((sig, cost, Qs))
                continue

            for q in range(0, sig[1] + 1):
                new_sig = extend(sig, q, D)
                stack.append((new_sig, new_cost, Qs + [q]))
        #best = min(possible_res, key=lambda x: calc_cost(text, generate_tree(color_sizes, x[2])[1]))
        best = min(possible_res, key=lambda x : x[1])
        return best[2]



class Node:
    def __init__(self, level, parent, number=0):
        self.number = number
        self.level = level
        self.parent = parent
        self.children = []

    def id(self):
        if not self.parent:
            return ""
        return f"{self.parent.id()} {self.number}"

    def visualize(self, tree):
        tree.create_node(self.id(), self.id(), parent=self.parent.id() if self.parent else None)
        for c in self.children:
            c.visualize(tree)


def parse_file(filename):
    with open(filename, "r") as f:
        _, color_sizes, text = f.readlines()
        color_sizes = list(map(int, color_sizes.split()))
    return color_sizes, text

def get_frequencies(text):
    return {i : text.count(i) / len(text) for i in set(text)}

if __name__ == '__main__':

    filename = "Examples/schmuck8.txt"
    color_sizes, text = parse_file(filename)
    encoder = Encoder(get_frequencies(text), color_sizes)

    Qs = encoder.encode_bfs()
    root, codes = (generate_tree(color_sizes, Qs))


    #graphviz.render("dot", "png","test.dot")
    occurrences = [text.count(i) for i in set(text)]
    occurrences = sorted(occurrences, reverse=True)
    costs = sorted(i[1] for i in codes)
    total_cost = 0
    for i in range(len(occurrences)):
        total_cost += occurrences[i] * costs[i]
    print(total_cost)
    print(total_cost / len(text))

    # TODO des ist vlt op: https://en.wikipedia.org/wiki/Asymmetric_numeral_systems
