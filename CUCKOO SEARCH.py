
import numpy as np
import matplotlib.pyplot as plt

start = np.array([0, 0])
goal = np.array([10, 10])

obstacles = [[2, 2, 4, 6], [5, 4, 7, 8], [7, 0, 8, 3]]

n = 20
Pa = 0.25
T = 100
alpha = 0.2
points = 5

def generate():
    return np.random.uniform(0, 10, (points, 2))

def fitness(path):
    p = np.vstack([start, path, goal])
    d = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    penalty = 0

    for a, b in zip(p[:-1], p[1:]):
        for t in np.linspace(0, 1, 30):
            x, y = a + t * (b - a)
            for x1, y1, x2, y2 in obstacles:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    penalty += 10000
                    break
            if penalty:
                break

    return d + penalty

nests = np.array([generate() for _ in range(n)])
values = np.array([fitness(x) for x in nests])

best = nests[np.argmin(values)].copy()
best_value = min(values)
history = []

for t in range(T):
    for i in range(n):
        new = np.clip(nests[i] + alpha * np.random.standard_cauchy((points, 2)), 0, 10)
        f = fitness(new)
        j = np.random.randint(n)

        if f < values[j]:
            nests[j] = new
            values[j] = f

    for i in range(n):
        if np.random.rand() < Pa:
            nests[i] = generate()
            values[i] = fitness(nests[i])

    j = np.argmin(values)

    if values[j] < best_value:
        best = nests[j].copy()
        best_value = values[j]

    history.append(best_value)

path = np.vstack([start, best, goal])

print("Best Path:\n", path)
print("Best Fitness:", best_value)
print("Collision Free:", fitness(best) < 10000 + np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))

for x1, y1, x2, y2 in obstacles:
    plt.fill([x1, x2, x2, x1], [y1, y1, y2, y2], 'red')

plt.plot(path[:, 0], path[:, 1], 'b-o')
plt.scatter(*start, color='green', label='Start')
plt.scatter(*goal, color='purple', label='Goal')
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.grid()
plt.title("Robot Path Planning using Cuckoo Search")
plt.show()

plt.plot(history)
plt.xlabel("Iterations")
plt.ylabel("Best Fitness")
plt.title("Convergence Graph")
plt.grid()
plt.show()
