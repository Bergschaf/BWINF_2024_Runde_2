import time
from collections import defaultdict
import rust_encoder
import math

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
            c_copy.parent = new_node
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


def generate_tree(Qs: list[int], color_sizes: list[int]):
    """
    Generiert einen vollständigen Baum
    :param Qs: Anzahl an internen Knoten pro Ebene (Werte von q, mit denen Erweitert wurde)
    :param color_sizes: Größen der Perlen
    :return:
    """
    color_sizes = sorted(color_sizes)
    root = Node(None, 0)
    Levels = defaultdict(list)
    Levels[0].append(root)
    for i, c in enumerate(color_sizes):
        # Die neuen Kinder werden anhand der Position der Perlengröße in der color_sizes Liste nummeriert
        root.children.append(Node(root, i))
        Levels[c].append(root.children[-1])

    for i, q in enumerate(Qs):
        if q == 0:
            continue
        for j, n in enumerate(Levels[i + 1]):
            for ii, c in enumerate(color_sizes):
                # Die neuen Kinder werden anhand der Position der Perlengröße in der color_sizes Liste nummeriert
                n.children.append(Node(n, ii))
                Levels[c + i + 1].append(n.children[-1])
            if j + 1 == q:
                break
    return root


def get_leaves_rec(node):
    """
    Gibt eine Liste alle Blätter zurück, die Kinder dieses Knotens sind
    """
    if not node.children:
        return [node]
    else:
        a = []
        for c in node.children:
            a.extend(get_leaves_rec(c))
        return a


def get_nodes_rec(node):
    """
    Gibt eine Liste aller Knoten zurück, die Kinder dieses Knotens sind
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
    Gibt alle Codekombinationen zurück, die die Blätter dieses Baums ausdrücken
    :returns: [(code, cost), ...]
    """
    leaves = get_leaves_rec(root)
    # get codes
    codes = []
    for l in leaves:
        code = []
        # Der Code besteht aus den nummern der Knoten auf dem Pfad von der Wurzel des Baums zu dem entsprechenden Blatt
        while l.parent:
            code.append(l.number)
            l = l.parent
        cost = 0
        for c in code:
            cost += color_sizes[c]
        # Der code wird hier umgedreht, da er sonst von der Wurzel zu dem Blatt gehen würde
        codes.append((code[::-1], cost))
    return codes


def attach_tree_at_leaves(main_root, sub_root, fixed_slots, color_sizes):
    """
    Hängt an jedes Blatt des Hauptbaums eine Kopie von sub_root an
    :param fixed_slots: Anzahl der obersten Blätter die nicht ersetzt werden sollen,
                        da sie zu häufigen Buchstaben gehören
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


def calc_cost(text, codes):
    """
    Berechnet die Kosten einer Liste an Codes, wenn sie optimal zu den Buchstaben zugewiesen werden
    """
    # Die Häufigkeiten werden absteigend sortiert
    occurrences = [text.count(i) for i in set(text)]
    occurrences = sorted(occurrences, reverse=True)

    # Die Kosten werden aufsteigend sortiert
    costs = sorted(i[1] for i in codes)

    total_cost = 0
    for i in range(len(occurrences)):
        # Da die Häufigkeiten absteigend und die Kosten aufsteigend sortiert sind, ist diese Zuweisung optimal
        total_cost += occurrences[i] * costs[i]
    return total_cost


def reduce(sig, n):
    """
    Wendet die Reduktionsregel auf eine Signatur an.
    (Alle Blätter, die nicht zu den n höchsten Blättern gehören werden abgeschnitten.)
    """
    new_sig = [min(sig[0], n)]
    for i in range(1, len(sig)):
        new_sig.append(min(sig[i], n - sum(new_sig)))
    return new_sig


def extend(sig, q, D):
    """
    Erweitert die Signatur um q, indem q interne Knoten auf Ebene i+1 zu internen Knoten werden
    """
    new_sig = [sig[0] + sig[1]] + sig[2:] + [0]
    x = ([-1] + D)
    for i in range(len(new_sig)):
        new_sig[i] = new_sig[i] + q * x[i]
    return new_sig


class Encoder:
    """
    Enthält verschiedene Methoden, um eine Codetabelle für die gegebenen Frequenzen zu erstellen.
    """

    def __init__(self, frequencies, color_sizes):
        self.frequencies = frequencies
        self.num_colors = len(color_sizes)
        self.color_sizes = sorted(color_sizes)

    def encode_naive(self):
        """
        Alle vollständigen Binärbäume mit m ≤ n * (r - 1) Blättern werden durchsucht
        """
        return generate_code_from_tree(self.get_tree_naive(self.frequencies.values()), self.color_sizes)

    def get_tree_naive(self, frequencies):
        """
        Alle vollständigen Binärbäume mit m ≤ n * (r - 1) Blättern werden durchsucht
        """
        # Frequenzen für jeden Buchstaben
        p = sorted(frequencies, reverse=True)
        # Die Anzahl der verschiedenen Buchstaben im Alphabet
        n = len(p)
        # Die größte Perle
        C = max(self.color_sizes)
        # Anzahl, wie oft jede Perlengröße vorkommt
        D = [self.color_sizes.count(i) for i in range(1, C + 1)]
        # Anzahl der verschiedenen Perlen
        r = len(self.color_sizes)

        print(f"N: {n}, C : {C}")
        # Speichert die Signatur, die Kosten und die Anzahl der Blätter, die gerade bearbeitet werden
        stack = [([0] + D, 0, [])]  # [(sig, cost, Qs (Expansions)), ...]

        # Der bisher beste Baum
        best_cost = float("inf")
        best_tree = None

        while stack:
            sig, cost, Qs = stack.pop()

            if sum(sig) > n * (r - 1):
                # Der Baum hat zu viele Blätter, um Fill(T_opt) zu sein
                continue

            if sig[0] >= n:
                # Der Baum hat genug Blätter, um alle Buchstaben zu codieren
                if cost < best_cost:
                    best_cost = cost
                    best_tree = Qs
                continue

            # Die neuen Kosten des Baums
            new_cost = cost + sum([p[i] for i in range(sig[0], n)])
            if new_cost > best_cost:
                # Der Baum ist zu teuer
                continue

            for q in range(0, sig[1] + 1):
                # Der Baum wird für jedes q erweitert
                new_sig = extend(sig, q, D)
                stack.append((new_sig, new_cost, Qs + [q]))

        # Der beste Baum wird anhand der Anzahlen an inneren Knoten auf jeder Ebene (Werte von q) generiert
        return generate_tree(best_tree, self.color_sizes)

    def encode_reduce(self):
        # Alle Bäume mit m ≤ n Blättern werden durchsucht
        return generate_code_from_tree(self.get_tree_reduce(self.frequencies.values()), self.color_sizes)

    def get_tree_reduce(self, frequencies):
        # Probability for each character of the alphabet
        # Frequenzen für jeden Buchstaben
        p = sorted(frequencies, reverse=True)
        # Die Anzahl der verschiedenen Buchstaben im Alphabet
        n = len(p)
        # Die größte Perle
        C = max(self.color_sizes)
        # Anzahl, wie oft jede Perlengröße vorkommt
        D = [self.color_sizes.count(i) for i in range(1, C + 1)]

        print(f"Encode Reduce: N: {n}, C : {C}")

        print(f"N: {n}, C : {C}")
        # Speichert die Signatur, die Kosten und die Anzahl der Blätter, die gerade bearbeitet werden
        stack = [([0] + D, 0, [])]  # [(sig, cost, Qs (Expansions)), ...]

        best_cost = float("inf")
        best_tree = None

        while stack:
            sig, cost, Qs = stack.pop()

            if sig[0] >= n:
                # Der Baum hat genug Blätter, um alle Buchstaben zu codieren
                if cost < best_cost:
                    best_cost = cost
                    best_tree = Qs
                continue

            new_cost = cost + sum([p[i] for i in range(sig[0], n)])
            if new_cost > best_cost:
                continue

            for q in range(0, min(sig[1], n - sum(sig[i] for i in range(self.color_sizes[1] + 1))) + 1):
                # Der Baum wird für jedes q erweitert (die erweiterung ist hier angewandt, d.h. es werden keine
                # "unnötigen" Werte von q durchlaufen

                # Der Baum wird erweitert und danach auf n Blätter reduziert
                new_sig = reduce(extend(sig, q, D), n)
                stack.append((new_sig, new_cost, Qs + [q]))

        # Der beste Baum wird anhand der Anzahlen an inneren Knoten auf jeder Ebene (Werte von q) generiert
        return generate_tree(best_tree, self.color_sizes)

    def get_tree_rust(self, frequencies, silent=True):
        """
        Die get_tree_reduce Funktion in Rust implementiert
        """
        qs = rust_encoder.encode_optimal_py(frequencies, self.color_sizes, silent)
        return generate_tree(qs, color_sizes)

    def encode_rust(self, silent=True):
        """
        Die encode_reduce Funktion in Rust implementiert
        """
        return generate_code_from_tree(self.get_tree_rust(list(self.frequencies.values()), silent), self.color_sizes)

    def heuristic(self, chunk_size=50):
        cutoff = 0.05  # Alle Buchstaben, die häufiger vorkommen, sind häufige Buchstaben

        # Die häufigen Buchstaben
        common = [i for i in self.frequencies.values() if i > cutoff]

        # Die seltenen Buchstaben
        not_common = sorted(i for i in self.frequencies.values() if i <= cutoff)

        # Die seltenen Buchstaben werden in chunks aufgeteilt
        chunks = []
        for i in range((len(not_common) // chunk_size)):
            chunks.append(not_common[i * chunk_size: (i + 1) * chunk_size])
        chunks.append(not_common[(len(not_common) // chunk_size) * chunk_size:])

        # Für den ersten Chunk wird ein optimaler Baum generiert
        chunk_root = self.get_tree_rust(chunks[0])

        # Für die häufigen Buchstaben und die Summen der Chunks wird ein Baum generiert
        main_root = self.get_tree_rust(common + [sum(c) for c in chunks])

        # Der chunk_root Baum wird an die passenden Blätter des main_root Baums angehängt
        main_root = attach_tree_at_leaves(main_root, chunk_root, len(common), color_sizes)

        # Der code wird anhand des fertigen Baums generiert
        return generate_code_from_tree(main_root, self.color_sizes)

    def optimize_heuristic(self, text):
        """
        Versucht einen möglichst guten Baum mit der heuristik zu finden, indem die chunk_size optimiert wird
        """
        n = len(self.frequencies)
        max_leaves = 40
        min_chunk_size = n // max_leaves + 1
        best_cost = float("inf")
        best_code = None
        for i in range(min_chunk_size, max_leaves):
            codes = self.heuristic(i)
            cost = calc_cost(text, codes)
            if cost < best_cost:
                best_cost = cost
                best_code = codes
        return best_code


def parse_file(filename):
    """
    Liest eine Beispieldatei ein
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        color_sizes = lines[1]
        text = lines[2][:-1]  # remove \n
        color_sizes = list(map(int, color_sizes.split()))
    return color_sizes, text


def get_frequencies(text):
    """
    Gibt die Frequenzen der Buchstaben in dem Text zurück
    """
    return {i: text.count(i) / len(text) for i in set(text)}


def print_code(code, text, color_sizes):
    """
    Gibt den Code und die Kosten aus
    """
    code = sorted(code, key=lambda x: x[1])
    print("Größe der Perlen: ", color_sizes)
    print()
    print("Ausgabeformat:\nBuchstabe(Häufigkeit): Code (Größe des Codes)")
    letters = sorted((i for i in set(text)), key=lambda x: text.count(x), reverse=True)
    for letter, c in zip(letters, code):
        print(f"\"{letter}\"({text.count(letter)}):\t {c[0]} ({c[1]})")
    print("----------------------------------------")
    print(f"Gesamtlänge der Perlenkette: {calc_cost(text, code)}")
    print()


if __name__ == '__main__':
    should_print_code = False
    for i in range(10):
        i = 0

        filename = f"Examples/schmuck{i}.txt"
        color_sizes, text = parse_file(filename)
        encoder = Encoder(get_frequencies(text), color_sizes)
        start = time.time()
        if i == 9:
            code = encoder.optimize_heuristic(text)
        else:
            code = encoder.encode_rust(silent=False)
        end = time.time()
        print(f"Beispiel: {filename}")
        print(f"Dauer: {(end - start) * 1000:.2f} ms")
        if should_print_code:
            print_code(code, text, color_sizes)
        else:
            print(f"Gesamtlänge der Perlenkette: {calc_cost(text, code)}")
            print(f"n = {len(set(text))}, c = {len(color_sizes)}")
            n = len(set(text))
            C = len(color_sizes)
            print(f"would have visited: {math.comb(n + C + 1, C + 1)}")
        exit()