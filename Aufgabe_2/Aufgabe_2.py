import numpy as np
from PIL import Image

# TODO beweis, dass das NP-schwer ist?


filename = "Examples/labyrinthe5.txt"


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

    def visualize(self, path, filename="test.png"):
        img = np.ones((self.m * 20 + 1, self.n * 20, 3), dtype=np.uint8) * 255
        def draw_rectangle(img, x, y, widht, height, color=(0, 255, 0)):
            for i in range(x, x + widht):
                for j in range(y, y + height):
                    if i < img.shape[1] and j < img.shape[0]:
                        img[j, i] = color
            return img

        def draw_line(img, x0, y0, x1, y1, color=(0, 0, 0)):
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) > abs(dy):
                if x0 > x1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0
                for x in range(x0, x1):
                    y = y0 + dy * (x - x0) // dx
                    if y < img.shape[0] and y >= 0 and x < img.shape[1] and x >= 0:
                        img[y, x] = color
            else:
                if y0 > y1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0
                for y in range(y0, y1):
                    x = x0 + dx * (y - y0) // dy
                    if y < img.shape[0] and y >= 0 and x < img.shape[1] and x >= 0:
                        img[y, x] = color
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
        for ii, (i, j) in enumerate(path):
            img = draw_rectangle(img, i * 20 + 5, j * 20 + 5, 10, 10, (0, 0, 255))
            # draw line
            if ii > 0:
                i0, j0 = path[ii - 1]
                img = draw_line(img, i0 * 20 + 10, j0 * 20 + 10, i * 20 + 10, j * 20 + 10, (0, 0, 255))

        img = Image.fromarray(img)
        img.save(filename)


    def get_reachable_neighbours(self, x, y):
        possible = []

        # UP
        if y > 0 and self.horizontal_walls[y - 1][x] == 0:
            possible.append((x, y - 1))

        # DOWN
        if y < self.m - 1 and self.horizontal_walls[y][x] == 0:
            possible.append((x, y + 1))

        # LEFT
        if x > 0 and self.vertical_walls[y][x - 1] == 0:
            possible.append((x - 1, y))

        # RIGHT
        if x < self.n - 1 and self.vertical_walls[y][x] == 0:
            possible.append((x + 1, y))

        for p in possible:
            if p in self.holes:
                possible.remove(p)
                # TODO wahrscheinlich nicht optimal
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
        path = [end]
        current = visited[end]
        while current:
            path.append(current)
            current = visited[current]
        return path


def main(filename):
    with open(filename, "r") as f:
        data = f.readlines()
        n, m = map(int, data.pop(0).split())
        l1 = Labyrinth(n, m)
        offset = l1.parse(data)


        l1.visualize([])
        l2 = Labyrinth(n, m)
        l2.parse(data[offset:])
        l2.visualize(l2.get_best_path_dijkstra(), "test2.png")
        l1.visualize(l1.get_best_path_dijkstra(), "test1.png")
if __name__ == '__main__':
    main(filename)