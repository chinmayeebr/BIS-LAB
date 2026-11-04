
import random
import math

cities = [
    (10,20), (15,25), (20,15), (25,30), (30,20),
    (35,35), (40,15), (45,25), (50,40), (55,20),
    (60,35), (65,15), (70,30), (75,45), (80,20),
    (85,35), (90,15), (95,40), (100,25), (105,45)
]

m, alpha, beta, rho, Q, T = 10, 1, 2, 0.5, 100, 100

def distance(a, b):
    return math.dist(a, b)

def kmeans(points, k):
    centers = random.sample(points, k)

    for _ in range(20):
        clusters = [[] for _ in range(k)]

        for p in points:
            c = min(range(k), key=lambda i: distance(p, centers[i]))
            clusters[c].append(p)

        for i in range(k):
            if clusters[i]:
                centers[i] = (
                    sum(p[0] for p in clusters[i]) / len(clusters[i]),
                    sum(p[1] for p in clusters[i]) / len(clusters[i])
                )

    return clusters, centers

def aco(points):
    n = len(points)

    if n <= 1:
        return list(range(n)), 0

    d = [[distance(points[i], points[j]) for j in range(n)] for i in range(n)]
    tau = [[1.0] * n for _ in range(n)]

    best_route = None
    best_dist = float('inf')

    for _ in range(T):
        routes = []

        for _ in range(m):
            start = random.randrange(n)
            route = [start]
            unvisited = set(range(n)) - {start}

            while unvisited:
                i = route[-1]
                choices = sorted(unvisited)

                weights = [
                    tau[i][j] ** alpha * (1 / d[i][j]) ** beta
                    for j in choices
                ]

                total = sum(weights)
                probs = [w / total for w in weights]

                j = random.choices(choices, weights=probs)[0]
                route.append(j)
                unvisited.remove(j)

            route.append(start)

            dist = sum(d[route[i]][route[i+1]] for i in range(n))
            routes.append((route, dist))

            if dist < best_dist:
                best_dist = dist
                best_route = route[:]

        for i in range(n):
            for j in range(n):
                tau[i][j] *= (1-rho)

        for route, dist in routes:
            for i in range(n):
                a, b = route[i], route[i+1]
                tau[a][b] += Q / dist
                tau[b][a] += Q / dist

    return best_route, best_dist

clusters, centers = kmeans(cities, 4)

print("K-MEANS CLUSTERS")

for i, c in enumerate(clusters):
    print("Cluster", i+1, ":", c)

cluster_order, _ = aco(centers)

final_route = []

for c in cluster_order[:-1]:
    points = clusters[c]
    route, _ = aco(points)
    local_route = [points[i] for i in route[:-1]]

    if final_route:
        if distance(final_route[-1], local_route[-1]) < distance(final_route[-1], local_route[0]):
            local_route.reverse()

    final_route.extend(local_route)

final_route.append(final_route[0])

total = sum(distance(final_route[i], final_route[i+1])
            for i in range(len(final_route)-1))

print("\nCLUSTER ORDER:", [i+1 for i in cluster_order[:-1]])
print("\nBEST ROUTE:")
print(final_route)
print("\nTOTAL DISTANCE:", round(total, 2))
