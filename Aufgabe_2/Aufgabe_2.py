import numpy as np
from PIL import Image

# TODO beweis, dass das NP-schwer ist?


filename = "Examples/labyrinthe1.txt"


class Labyrinth:
    def __init__(self, n, m):
        self.vertical_walls = []
        self.horizontal_walls = []
        self.holes = []
        self.n = n
        self.m = m

    def parse(self, data) -> int:
        """

        returns the number of lines parsed
        """
        n, m = self.n, self.m
        #vertical_walls = []  # if there is a wall to the right of (i,j), then vertical_walls[i][j] = 1
        #horizontal_walls = []  # if there is a wall below (i,j), then horizontal_walls[i][j] = 1
        #holes = []  # list of holes

        for m_ in range(m):
            self.vertical_walls.append(list(map(int, data[m_].split())))

        for m_ in range(m - 1):
            self.horizontal_walls.append(list(map(int, data[m + m_].split())))

        num_holes = int(data[2 * m - 1])
        for n in range(num_holes):
            self.holes.append(list(map(int, data[2 * m + n].split())))
        return 2 * m + num_holes

    def visualize(self, path):
        img = np.ones((self.n * 20 + 1, self.m * 20, 3), dtype=np.uint8) * 255
        def draw_rectangle(img, x, y, widht, height, color=(0, 255, 0)):
            for i in range(x, x + widht):
                for j in range(y, y + height):
                    if i < img.shape[1] and j < img.shape[0]:
                        img[j, i] = color
            return img

        # Draw vertical walls
        for j in range(self.m):
            for i in range(self.n-1):
                if self.vertical_walls[j][i] == 1:
                    img = draw_rectangle(img, (i+1) * 20 - 3, j * 20, 5, 20)

        # Draw horizontal walls
        for m_ in range(self.m - 1):
            for i in range(self.n):
                if self.horizontal_walls[m_][i] == 1:
                    img = draw_rectangle(img, i * 20, (m_ + 1) * 20 - 3, 20, 5)

        # Draw holes
        for hole in self.holes:
            img = draw_rectangle(img, hole[0] * 20, hole[1] * 20,20, 20, (255, 0, 0))

        # Draw path
        for i, j in path:
            img = draw_rectangle(img, i * 20 + 5, j * 20 + 5, 10, 10, (0, 0, 255))

        img = Image.fromarray(img)
        img.save("test.png")

    def get_reachable_neighbours(self, i, j):
        possible = []
        # Up
        if i > 0 and not self.horizontal_walls[i - 1][j]:
            possible.append((i - 1, j))
        # Down
        if i < self.n - 1 and not self.horizontal_walls[i][j]:
            possible.append((i + 1, j))

        # Left
        if j > 0 and not self.vertical_walls[i][j - 1]:
            possible.append((i, j - 1))
        # Right
        if j < self.m - 1 and not self.vertical_walls[i][j]:
            possible.append((i, j + 1))

        for p in possible:
            if p in self.holes:
                possible.remove(p)
                # TODO probably not optimal

        return possible

    def get_best_path_dijkstra(self):
        start = (0, 0)
        end = (self.n - 1, self.m - 1)
        stack = [(start, 0, None)]
        visited = {start : None}
        while stack:
            stack = sorted(stack, key=lambda x: x[1])
            current, dist, previous = stack.pop(0)

            visited[current] = previous
            if current == end:
                break
            for neighbour in self.get_reachable_neighbours(*current):
                if neighbour not in visited.keys():
                    for i in range(len(stack)):
                        if neighbour == stack[i][0]:
                            if dist + 1 < stack[i][1]:
                                stack[i] = (neighbour, dist + 1, current)
                            break
                    else:
                        stack.append((neighbour, dist + 1, current))
        path = []
        current = visited[end]
        while current:
            path.append(current)
            current = visited[current]
        return path[::-1]


def main(filename):
    with open(filename, "r") as f:
        data = f.readlines()
        n, m = map(int, data.pop(0).split())
        l1 = Labyrinth(n, m)
        offset = l1.parse(data)

        l2 = Labyrinth(n, m)
        l2.parse(data[offset:])

        l1.visualize(l1.get_best_path_dijkstra())
if __name__ == '__main__':
    main(filename)