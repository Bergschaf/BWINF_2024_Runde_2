from readline import write_history_file
from numba import jit
import numpy as np
from PIL import Image
from multiprocessing import Pool
# TODO beweis, dass das NP-schwer ist?


class Instruction:
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"

    @staticmethod
    @jit
    def walk(x, y, instruction):
        if instruction == "UP":
            return x, y - 1
        if instruction == "DOWN":
            return x, y + 1
        if instruction == "LEFT":
            return x - 1, y
        if instruction == "RIGHT":
            return x + 1, y
        raise ValueError("Invalid instruction")

    @staticmethod
    def inv(instruction):
        if instruction == Instruction.UP:
            return Instruction.DOWN
        if instruction == Instruction.DOWN:
            return Instruction.UP
        if instruction == Instruction.LEFT:
            return Instruction.RIGHT
        if instruction == Instruction.RIGHT:
            return Instruction.LEFT
        raise ValueError("Invalid instruction")


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
        # vertical_walls = []  # if there is a wall to the right of (i,j), then vertical_walls[i][j] = 1
        # horizontal_walls = []  # if there is a wall below (i,j), then horizontal_walls[i][j] = 1
        # holes = []  # list of holes

        for m_ in range(m):
            self.vertical_walls.append(list(map(int, data[m_].split())))

        for m_ in range(m - 1):
            self.horizontal_walls.append(list(map(int, data[m + m_].split())))

        num_holes = int(data[2 * m - 1])
        for n in range(num_holes):
            self.holes.append(tuple(map(int, data[2 * m + n].split())))
        self.holes = set(self.holes)
        return 2 * m + num_holes

    def visualize(self, path, filename="test.png"):
        img = np.ones((self.m * 20 + 1, self.n * 20, 3), dtype=np.uint8) * 0  # 255

        def draw_rectangle(img, x, y, widht, height, color=(0, 255, 0), alpha=0.3):
            color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            for i in range(x, x + widht):
                for j in range(y, y + height):
                    if i < img.shape[1] and j < img.shape[0]:
                        img[j, i] = color + img[j, i]
            return img

        def draw_line(img, x0, y0, x1, y1, color=(0, 0, 0), alpha=0.3):
            color = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
            dx = x1 - x0
            dy = y1 - y0
            if abs(dx) > abs(dy):
                if x0 > x1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0
                for x in range(x0, x1):
                    y = y0 + dy * (x - x0) // dx
                    if y < img.shape[0] and y >= 0 and x < img.shape[1] and x >= 0:
                        img[y, x] = color + img[y, x]
            else:
                if y0 > y1:
                    x0, x1 = x1, x0
                    y0, y1 = y1, y0
                for y in range(y0, y1):
                    x = x0 + dx * (y - y0) // dy
                    if y < img.shape[0] and y >= 0 and x < img.shape[1] and x >= 0:
                        img[y, x] = color + img[y, x]
            return img

        # Draw vertical walls
        for j in range(self.m):
            for i in range(self.n - 1):
                if self.vertical_walls[j][i] == 1:
                    img = draw_rectangle(img, (i + 1) * 20 - 3, j * 20, 5, 20, alpha=0.4)

        # Draw horizontal walls
        for m_ in range(self.m - 1):
            for i in range(self.n):
                if self.horizontal_walls[m_][i] == 1:
                    img = draw_rectangle(img, i * 20, (m_ + 1) * 20 - 3, 20, 5, alpha=0.4)

        # Draw holes
        for hole in self.holes:
            img = draw_rectangle(img, hole[0] * 20, hole[1] * 20, 20, 20, (255, 0, 0))

        # Draw path
        for ii, (i, j) in enumerate(path):
            img = draw_rectangle(img, i * 20 + 5, j * 20 + 5, 10, 10, (0, 0, 255))
            # draw line
            if ii > 0:
                i0, j0 = path[ii - 1]
                img = draw_line(img, i0 * 20 + 10, j0 * 20 + 10, i * 20 + 10, j * 20 + 10, (0, 0, 255))

        img = Image.fromarray(img)
        img.save(filename)

    def is_wall_below(self, x, y):
        if y == self.m - 1:
            return True
        return self.horizontal_walls[y][x] == 1

    def is_wall_above(self, x, y):
        if y == 0:
            return True
        return self.horizontal_walls[y - 1][x] == 1

    def is_wall_right(self, x, y):
        if x == self.n - 1:
            return True
        return self.vertical_walls[y][x] == 1

    def is_wall_left(self, x, y):
        if x == 0:
            return True
        return self.vertical_walls[y][x - 1] == 1

    def get_reachable_neighbours(self, x, y):
        possible = []

        if not self.is_wall_below(x, y):
            possible.append((x, y + 1))
        if not self.is_wall_above(x, y):
            possible.append((x, y - 1))
        if not self.is_wall_right(x, y):
            possible.append((x + 1, y))
        if not self.is_wall_left(x, y):
            possible.append((x - 1, y))
        for p in possible:
            if p == (18, 6):
                print("test")
            if p in self.holes:
                possible.remove(p)
                # TODO wahrscheinlich nicht optimal
        return possible

    def get_best_path_dijkstra(self):
        start = (0, 0)
        end = (self.n - 1, self.m - 1)
        stack = [(start, 0, None)]
        visited = {start: None}
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

        if end not in visited:
            raise Exception("No path found")

        path = [end]
        current = visited[end]
        while current:
            path.append(current)
            current = visited[current]
        return path[::-1]

    def get_instructions_from_path(self, path):
        """
        :return: [(instruction, [every instruction that would go against a wall])]
        TODO dass muss eig nicht sein, dass kann direkt während der Suche passieren
        """
        instructions = []
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            impossible = []
            if self.is_wall_left(x0, y0):
                impossible.append(Instruction.LEFT)
            if self.is_wall_right(x0, y0):
                impossible.append(Instruction.RIGHT)
            if self.is_wall_above(x0, y0):
                impossible.append(Instruction.UP)
            if self.is_wall_below(x0, y0):
                impossible.append(Instruction.DOWN)

            x1, y1 = path[i]
            if x1 > x0:
                instructions.append((Instruction.RIGHT, impossible))
            elif x1 < x0:
                instructions.append((Instruction.LEFT, impossible))
            elif y1 > y0:
                instructions.append((Instruction.DOWN, impossible))
            elif y1 < y0:
                instructions.append((Instruction.UP, impossible))

        return instructions

    def is_wall_in_direction(self, x, y, instruction):
        if instruction == Instruction.UP:
            return self.is_wall_above(x, y)
        elif instruction == Instruction.DOWN:
            return self.is_wall_below(x, y)
        elif instruction == Instruction.LEFT:
            return self.is_wall_left(x, y)
        elif instruction == Instruction.RIGHT:
            return self.is_wall_right(x, y)
        raise ValueError("Invalid instruction")

    def move(self, x, y, instruction):
        if self.is_wall_in_direction(x, y, instruction):
            return (x, y)

        if x == self.n - 1 and y == self.m - 1:
            return (x, y)
        new_pos = Instruction.walk(x, y, instruction)
        if new_pos in self.holes:
            return (0, 0)
        return new_pos

    def move_without_fix_at_end(self, x, y, instruction):
        if self.is_wall_in_direction(x, y, instruction):
            return (x, y)

        new_pos = Instruction.walk(x, y, instruction)
        if new_pos in self.holes:
            return (0, 0)
        return new_pos


#    def move(self, x, y, instruction):
#        if x == self.n -1 and y == self.m - 1:
#            return (x, y)
#        match instruction:
#            case Instruction.UP:
#                if self.is_wall_above(x, y):
#                    return (x, y)
#            case Instruction.DOWN:
#                if self.is_wall_below(x, y):
#                    return (x, y)
#            case Instruction.LEFT:
#                if self.is_wall_left(x, y):
#                    return (x, y)
#            case Instruction.RIGHT:
#                if self.is_wall_right(x, y):
#                    return (x, y)
#
#        new_pos = Instruction.move(x, y, instruction)
#        if new_pos in self.holes:
#            return (0,0)
#        return new_pos


def get_double_path_bfs(l1, l2):
    start = ((0, 0), (0, 0))
    n, m = l1.n, l1.m
    assert n == l2.n and m == l2.m
    end = ((n - 1, m - 1), (n - 1, m - 1))

    # TODO ab jetzt wirds ineffizient
    stack = [(start, None, None)]
    visited = {start: None}
    count = 0

    while stack:
        if count % 100000 == 0:
            print(count)
        # stack = sorted(stack, key=lambda x: x[1])
        current, previous, instruction = stack.pop(0)

        if current == end:
            print("Found Path")

            break

        for direction in [Instruction.DOWN, Instruction.LEFT, Instruction.UP, Instruction.RIGHT]:
            new_pos = (l1.move(*current[0], direction), l2.move(*current[1], direction))
            if new_pos == current:
                continue

            if new_pos not in visited:
                stack.append((new_pos, current, direction))
                visited[new_pos] = current
                count += 1
    path1 = [end[0]]
    path2 = [end[1]]
    instructions = []
    current = visited[end]
    while current:
        # instructions.append(inst)
        path1.append(current[0])
        path2.append(current[1])
        current = visited[current]

    return path1[::-1], path2[::-1], instructions[::-1]


def forward(l1, l2, nodes, forward_visited, backward_visited):
    """
    Returns all the nodes one level deeper
    returns Nodes, Found Path
    """
    new_nodes = []  # TODO vlt as Set
    for node in nodes:
        for direction in [Instruction.DOWN, Instruction.LEFT, Instruction.UP, Instruction.RIGHT]:
            new_pos = (l1.move(*node[0], direction), l2.move(*node[1], direction))
            if new_pos == node:
                continue
            if new_pos not in forward_visited:
                new_nodes.append(new_pos)
                forward_visited[new_pos] = node
                if new_pos in backward_visited:
                    print("Found Path")
                    return new_nodes, new_pos

    return new_nodes, False

def backward(l1, l2, nodes, forward_visited, backward_visited):
    """
    Returns all the nodes that can reach a node in nodes in one step
    returns nodes, found path, can continue (false if one pos is (0,0), could be from every hole)
    """
    new_nodes = []  # TODO vlt as Set
    can_continue = True
    for node in nodes:
        # TODO parallelisieren
        for direction in [Instruction.DOWN, Instruction.LEFT, Instruction.UP, Instruction.RIGHT]:
            # TODO Mülltonne
            if l1.is_wall_in_direction(*node[0], Instruction.inv(direction)):
                if l2.is_wall_in_direction(*node[1], Instruction.inv(direction)):
                    to_process = [(node[0], l2.move_without_fix_at_end(*node[1], direction)),
                                  (l1.move_without_fix_at_end(*node[0], direction),
                                   l2.move_without_fix_at_end(*node[1], direction)),
                                  (l1.move_without_fix_at_end(*node[0], direction), node[1])]
                else:
                    to_process = [(node[0], l2.move_without_fix_at_end(*node[1], direction)),
                                  (l1.move_without_fix_at_end(*node[0], direction),
                                   l2.move_without_fix_at_end(*node[1], direction))]
            else:
                if l2.is_wall_in_direction(*node[1], Instruction.inv(direction)):
                    to_process = [(l1.move_without_fix_at_end(*node[0], direction), node[1]),
                                  (l1.move_without_fix_at_end(*node[0], direction),
                                   l2.move_without_fix_at_end(*node[1], direction))]
                else:
                    to_process = [(l1.move_without_fix_at_end(*node[0], direction),
                                   l2.move_without_fix_at_end(*node[1], direction))]
            for new_pos in to_process:
                if new_pos == node:
                    continue
                if new_pos not in backward_visited:
                    if (l1.move(*new_pos[0], Instruction.inv(direction)),
                        l2.move(*new_pos[1], Instruction.inv(direction))) == node:
                        new_nodes.append(new_pos)
                        backward_visited[new_pos] = node
                        if new_pos in forward_visited:
                            print("Found Path hier")
                            return new_nodes, new_pos, None

    return new_nodes, False, can_continue



def retrace_path_forward(forward_visited, connection):
    """
    :param l1:
    :param l2:
    :param forward_visited:
    :param connection: The node that connects the two paths
    :return:
    """
    current = forward_visited[connection]
    path1 = []
    path2 = []
    while current:
        path1.append(current[0])
        path2.append(current[1])
        current = forward_visited[current]
    return path1[::-1], path2[::-1]

def retrace_path_backward(backward_visited, connection):
    current = backward_visited[connection]
    path1 = []
    path2 = []
    while current:
        path1.append(current[0])
        path2.append(current[1])
        current = backward_visited[current]
    return path1, path2


def get_double_path_bidibfs(l1, l2):
    start = ((0, 0), (0, 0))
    n, m = l1.n, l1.m
    assert n == l2.n and m == l2.m
    end = ((n - 1, m - 1), (n - 1, m - 1))
    level = 0
    forward_stack = [start]
    backward_stack = [end]
    forward_visited = {start: None}
    backward_visited = {end: None}
    can_continue = True
    found_path = False
    while forward_stack and backward_stack:
        if level % 100 == 0:
            print(level, len(forward_stack), len(backward_stack), can_continue)
        if not can_continue or len(forward_stack) <= len(backward_stack):
            forward_stack, found_path = forward(l1, l2, forward_stack, forward_visited, backward_visited)
        else:
            backward_stack, found_path, can_continue = backward(l1, l2, backward_stack, forward_visited,
                                                                backward_visited)
        if found_path:
            break
        level += 1
    if not found_path:
        print("No path found")
        return False
    path1_1, path2_1 = retrace_path_forward(forward_visited, found_path)
    path1_2, path2_2 = retrace_path_backward(backward_visited, found_path)
    return path1_1 + [found_path[0]] + path1_2, path2_1 + [found_path[1]] + path2_2



def main(filename):
    with open(filename, "r") as f:
        data = f.readlines()
        n, m = map(int, data.pop(0).split())
        l1 = Labyrinth(n, m)
        offset = l1.parse(data)
        print(f"offset: {offset}")

        l2 = Labyrinth(n, m)
        l2.parse(data[offset:])
    l1.visualize([], filename="test1.png")
    l2.visualize([], filename="test2.png")
    # print(l2.get_best_path_dijkstra())
    #path1, path2, inst = get_double_path_bfs(l1, l2)
    #bfs_len = len(path1)
    path1, path2 = get_double_path_bidibfs(l1, l2)
    l1.visualize(path1, filename="test1.png")
    l2.visualize(path2, filename="test2.png")
    print(path1)
    print(path2)
    #print("BFS len " + str(bfs_len))

    print(f"Länge: {len(path2)}")


filename = "Examples/labyrinthe6.txt"

if __name__ == '__main__':
    main(filename)
