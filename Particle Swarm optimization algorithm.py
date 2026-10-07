import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = np.array(Image.open("input.jpg").convert("L"))

def fitness(t):
    t = np.sort(t.astype(int))
    hist = np.bincount(img.ravel(), minlength=256)
    p = hist / hist.sum()
    mean = np.sum(np.arange(256) * p)
    b = 0

    for l, u in zip([0] + list(t), list(t) + [256]):
        w = p[l:u].sum()
        if w:
            m = np.sum(np.arange(l, u) * p[l:u]) / w
            b += w * (m - mean) ** 2

    return b

n, d, it = 20, 3, 50
w, c1, c2 = 0.7, 1.5, 1.5

x = np.random.randint(1, 255, (n, d))
v = np.random.uniform(-10, 10, (n, d))

pbest = x.copy()
pfit = np.array([fitness(i) for i in x])

g = pbest[np.argmax(pfit)].copy()
gfit = max(pfit)

for _ in range(it):
    for i in range(n):
        r1, r2 = np.random.rand(d), np.random.rand(d)

        v[i] = (w * v[i] +
                c1 * r1 * (pbest[i] - x[i]) +
                c2 * r2 * (g - x[i]))

        x[i] = np.clip(x[i] + v[i], 1, 254)
        f = fitness(x[i])

        if f > pfit[i]:
            pfit[i] = f
            pbest[i] = x[i].copy()

        if f > gfit:
            gfit = f
            g = x[i].copy()

t = np.sort(g.astype(int))

seg = np.zeros_like(img)
seg[img > t[0]] = 85
seg[img > t[1]] = 170
seg[img > t[2]] = 255

print("Best Thresholds:", t)
print("Best Fitness:", gfit)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg, cmap="gray")
plt.title("PSO Segmentation")
plt.axis("off")

plt.show()
