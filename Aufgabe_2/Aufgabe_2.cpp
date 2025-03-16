#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include <queue>
#include <vector>

auto filename = "Examples/labyrinthe4.txt";
enum Instruction {UP, DOWN, LEFT, RIGHT};

struct Position {
    int x, y;



    bool operator<(const Position &other) const {
        return std::tie(x, y) < std::tie(other.x, other.y);
    }
};

bool operator==(const Position lhs, const Position rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

class Labyrinth {
public:
    int m, n;
    std::vector<std::vector<bool>> verticalWalls; // true if there is a wall to the right of (i,j)
    std::vector<std::vector<bool>> horizontalWalls; // true if there is a wall below (i,j)
    int num_holes; // true if there is a hole in the labyrinth
    std::vector<std::vector<bool>> holes; // true if there is a hole at (i,j)

    Labyrinth(int n, int m) {
        this->m = m;
        this->n = n;

    }

    void print() {
        // n, m
        std::cout << "n: " << n << " m: " << m << std::endl;

        // vertical walls
        std::cout << "Vertical Walls" << std::endl;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n-1; ++j) {
                std::cout << verticalWalls[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // horizontal walls
        std::cout << "Horizontal Walls" << std::endl;
        for (int i = 0; i < m - 1; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << horizontalWalls[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // num_holes
        std::cout << "Num Holes: " << num_holes << std::endl;
    }


    int parseFile (std::ifstream &file) {

        std::string line;

        verticalWalls.resize(m, std::vector<bool>(n-1, false));
        horizontalWalls.resize(m - 1, std::vector<bool>(n, false));

        // Read vertical walls
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n-1; ++j) {
                int wall;
                file >> wall;
                verticalWalls[i][j] = wall == 1;
            }
        }

        // Read horizontal walls
        for (int i = 0; i < m - 1; ++i) {
            for (int j = 0; j < n; ++j) {
                int wall;
                file >> wall;
                horizontalWalls[i][j] = wall == 1;
            }
        }

        // Read holes
        file >> num_holes;
        if (num_holes == 0) {
            return 2 * m;
        }
        holes.resize(m, std::vector<bool>(n, false));
        for (int i = 0; i < num_holes; ++i) {
            int x, y;
            file >> x >> y;
            holes[x][y] = true;
        }

        return 2 * m + num_holes;
    }

    bool is_wall_above(Position pos) {
        if (pos.y == 0) {
            return true;
        }
        return horizontalWalls[pos.y - 1][pos.x];
    }


    bool is_wall_below(Position pos) {
        if (pos.y == m - 1) {
            return true;
        }
        return horizontalWalls[pos.y][pos.x];
    }

    bool is_wall_left(Position pos) {
        if (pos.x == 0) {
            return true;
        }
        return verticalWalls[pos.y][pos.x - 1];
    }

    bool is_wall_right(Position pos) {
        if (pos.x == n - 1) {
            return true;
        }
        return verticalWalls[pos.y][pos.x];
    }

    Position check_hole(Position pos) {
        if (num_holes == 0) {
            return  pos;
        }

        if (holes[pos.y][pos.x]) {
            return {0, 0};
        }
        return pos;
    }

    Position move (Position pos, Instruction inst) {
        switch (inst) {
            case UP :
                if (!is_wall_above(pos)) {
                    return {pos.x, pos.y - 1};
                }
                return check_hole(pos);

            case DOWN : {
                if (!is_wall_below(pos)) {
                    return {pos.x, pos.y + 1};
                }
                return check_hole(pos);
            }
            case LEFT : {
                if (!is_wall_left(pos)) {
                    return {pos.x - 1, pos.y};
                }
                return check_hole(pos);
            }

            case RIGHT : {
                if (!is_wall_right(pos)) {
                    return {pos.x + 1, pos.y};
                }
                return check_hole(pos);
            }
            default: {
                std::cout << inst << std::endl;
                // raise error
                std::cout << "Invalid instruction" << std::endl;
                // abort
                throw std::invalid_argument("Invalid instruction");

            }
        }
    }
};


std::tuple<std::vector<Position>, std::vector<Position>, std::vector<Instruction>> get_double_path_bfs(Labyrinth &l1, Labyrinth &l2) {
    using State = std::pair<Position, Position>;

    State start = {{0, 0}, {0, 0}};
    int n = l1.n, m = l1.m;
    assert(n == l2.n && m == l2.m);
    State end = {{n - 1, m - 1}, {n - 1, m - 1}};

    std::queue<std::tuple<State, State, Instruction>> queue;
    std::map<State, State> visited;
    queue.emplace(start, start, UP);
    visited[start] = start;
    int count;

    while (!queue.empty()) {
        auto [current, previous, instruction] = queue.front();
        queue.pop();
        if (count % 10000 == 0) {
            std::cout << count << std::endl;
        }
        if (current == end) {
            std::cout << "Found Path" << std::endl;
            break;
        }

        for (Instruction direction : {DOWN, LEFT, UP, RIGHT}) {
            State new_pos = {l1.move(current.first, direction), l2.move(current.second, direction)};
            if (new_pos == current) {
                continue;
            }

            if (visited.find(new_pos) == visited.end()) {
                queue.push({new_pos, current, direction});
                visited[new_pos] = current;
                count++;
            }
        }
    }

    std::vector<Position> path1 = {end.first};
    std::vector<Position> path2 = {end.second};
    std::vector<Instruction> instructions;
    State current = end;

    while (!(current.first == start.first && current.second == start.second)) {
        State prev = visited[current];
        path1.push_back(prev.first);
        path2.push_back(prev.second);
        current = prev;
    }

    std::reverse(path1.begin(), path1.end());
    std::reverse(path2.begin(), path2.end());
    std::reverse(instructions.begin(), instructions.end());

    return {path1, path2, instructions};
}

void print_path(const std::vector<Position>& path) {
    for (auto &pos : path) {
        std::cout << "(" << pos.x << ", " << pos.y << ") ";
    }
    std::cout << std::endl;
}

int main() {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file " << filename << std::endl;
        return 1;
    }
    int m, n;
    file >> n >> m;



    Labyrinth labyrinth(n, m);
    labyrinth.parseFile(file);

    Labyrinth labyrinth2(n, m);
    labyrinth2.parseFile(file);

    auto [path1, path2, instructions] = get_double_path_bfs(labyrinth, labyrinth2);
    print_path(path1);
    print_path(path2);
    //labyrinth.print();
    //labyrinth2.print();

    file.close();

    return 0;
}