from PIL import Image
import numpy as np
filename = "Examples/labyrinthe6.txt"

def fill_rectangle(img, x, y, width, height, color=(0, 255, 0)):
    for i in range(x, x + width):
        for j in range(y, y + height):
            if i < img.shape[1] and j < img.shape[0]:
                img[j, i] = color
    return img

def visualize(file):
    with open(file, "r") as f:
        data = f.readlines()
        n, m = map(int, data.pop(0).split())
        # initialize the image white
        for j in range(2):
            img = np.ones((n * 20 + 1, m * 20, 3), dtype=np.uint8) * 255
            for l in range(m):
                line = list(map(int, data[l].split()))
                for i in range(n-1):
                    if line[i] == 1:
                        img = fill_rectangle(img, (i+1) * 20 - 3, l * 20, 5, 20)

            for m_ in range(m - 1):
                line = list(map(int, data[m + m_].split()))
                for i in range(n):
                    if line[i] == 1:
                        img = fill_rectangle(img, i * 20, (m_ + 1) * 20 - 3, 20, 5)
            num_holes = int(data[2 * m - 1])
            for n in range(num_holes):
                hole = list(map(int, data[2 * m + n].split()))
                img = fill_rectangle(img, hole[0] * 20, hole[1] * 20, 20, 20, (255, 0, 0))


            img = Image.fromarray(img)
            img.save(f"test_{j}.png")
            data = data[2 * m + num_holes:]



if __name__ == '__main__':
    visualize(filename)
