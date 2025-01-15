

class Encoder:
    def __init(self, text, num_colors, color_sizes):
        self.text = text
        self.num_colors = num_colors
        self.color_sizes = color_sizes

    def encode(self):
        # Huffmann
        # v  TODO rename, anzahl der einzelnen Buchstaben
        counts = {x : self.text.count(x) for x in set(self.text)}
        # TODO des ist vlt op: https://en.wikipedia.org/wiki/Asymmetric_numeral_systems