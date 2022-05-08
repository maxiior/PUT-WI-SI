import numpy as np
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import time


class LocalSearch:
    def __init__(self):
        self.coordinate_matrix = None
        self.distance_matrix = None

    def read_data(self, path):
        self.coordinate_matrix = np.genfromtxt(fname=path, delimiter=" ",
                                               skip_header=6, skip_footer=1, dtype=int)
        self._get_distance_matrix()

    def _get_distance_matrix(self):
        self.distance_matrix = np.empty(
            shape=(0, self.coordinate_matrix.shape[0]), dtype=int)

        for i in self.coordinate_matrix:
            distance_array = np.array([])
            for j in self.coordinate_matrix:
                distance = round(
                    math.sqrt(((j[1]-i[1])**2) + ((j[2]-i[2])**2)))
                distance_array = np.append(distance_array, distance)
            self.distance_matrix = np.vstack(
                [self.distance_matrix, distance_array])

    def make_visualizations(self, a, color):
        a += [a[0]]
        for i in range(len(a) - 1):
            plt.plot([self.coordinate_matrix[a[i]][1], self.coordinate_matrix[a[i+1]][1]],
                     [self.coordinate_matrix[a[i]][2], self.coordinate_matrix[a[i+1]][2]], color=color)

    def regret_heuristics(self, v1, random=False):
        vertexes = list(range(self.distance_matrix.shape[0]))
        vertexes.remove(v1)

        if not random:
            v2 = np.argmax(self.distance_matrix[v1, :])
        else:
            v2 = random.choice(vertexes)

        vertexes.remove(v2)
        cycles = [[v1], [v2]]

        while vertexes:
            for c in cycles:
                scores_matrix = np.empty((0, len(c)))
                for v in vertexes:
                    scores = np.array([])
                    for i in range(len(c)):
                        scores = np.append(
                            scores, self.distance_matrix[c[i - 1], v] + self.distance_matrix[v, c[i]] - self.distance_matrix[c[i - 1], c[i]])
                    scores_matrix = np.vstack([scores_matrix, scores])

                best_vertex = None
                regret = np.array([])

                if len(scores_matrix[0]) == 1:
                    best_vertex = vertexes[np.argmin(scores_matrix)]
                    c.insert(0, best_vertex)
                else:
                    sorted_array = np.sort(scores_matrix)[:, :2]
                    regret = np.array([i[1]-i[0] for i in sorted_array])
                    idx = np.argmax(regret - np.min(scores_matrix, axis=1))
                    best_vertex = vertexes[idx]
                    c.insert(np.argmin(scores_matrix[idx]), best_vertex)

                vertexes.remove(best_vertex)
        return cycles

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_scores(self, matrix):
        matrix += [matrix[0]]
        s1 = sum(self.distance_matrix[matrix[0][i], matrix[0][i+1]]
                 for i in range(len(matrix[0]) - 1))
        s2 = sum(self.distance_matrix[matrix[1][i], matrix[1][i+1]]
                 for i in range(len(matrix[1]) - 1))
        return s1 + s2

    def get_random_cycles(self):
        n = self.distance_matrix.shape[0]
        vertexes = list(range(n))
        random.shuffle(vertexes)
        return [vertexes[:n//2], vertexes[n//2:]]

    def find_vertex(self, c, i):
        try:
            return 0, c[0].index(i)
        except:
            try:
                return 1, c[1].index(i)
            except:
                return None, None

    def get_delta_and_move_of_swamping_vertex(self, cycles, ic1, ic2, i, j):
        c1, c2 = cycles[ic1], cycles[ic2]
        x1, y1, z1 = c1[(i-1) % len(c1)], c1[i], c1[(i+1) % len(c1)]
        x2, y2, z2 = c2[(j-1) % len(c2)], c2[j], c2[(j+1) % len(c2)]
        d = self.distance_matrix
        delta = d[x1, y2] + d[z1, y2] + d[x2, y1] + d[z2, y1] - \
            d[x1, y1] - d[z1, y1] - d[x2, y2] - d[z2, y2]
        move = 'swaping_vertex', delta, ic1, ic2, x1, y1, z1, x2, y2, z2
        return delta, move

    def get_delta_of_swamping_edge(self, a, b, c, d):
        return self.distance_matrix[a, c] + self.distance_matrix[b, d] - self.distance_matrix[a, b] - self.distance_matrix[c, d]

    def commit(self, cycles, m):
        if m[0] == 'swaping_edge':
            c, i = self.find_vertex(cycles, m[2])
            _, j = self.find_vertex(cycles, m[4])
            c = cycles[c]
            n = len(c)
            i = (i+1) % n
            j = (j-i) % n

            for k in range(abs(j)//2+1):
                a = (i+k) % n
                b = (i+j-k) % n
                c[a], c[b] = c[b], c[a]

        elif m[0] == 'swaping_vertex':
            i = cycles[m[2]].index(m[5])
            j = cycles[m[3]].index(m[8])
            cycles[m[2]][i], cycles[m[3]][j] = cycles[m[3]][j], cycles[m[2]][i]

    def generate_edges(self, n):
        return [(j, (j+i) % n) for j in range(n) for i in range(2, n-1)]

    def generate_vertices(self, n, m):
        return [(i, j) for i in range(n) for j in range(m)]

    def generate_starting_moves(self, cycles):
        moves = []
        for k in cycles:
            for i, j in self.generate_edges(len(k)):
                a, b, c, d = k[i], k[(i+1) % len(k)], k[j], k[(j+1) % len(k)]
                delta = self.get_delta_of_swamping_edge(a, b, c, d)
                if delta < 0:
                    moves.append(('swaping_edge', delta, a, b, c, d))

        for i, j in self.generate_vertices(len(cycles[0]), len(cycles[1])):
            delta, m = self.get_delta_and_move_of_swamping_vertex(
                cycles, 0, 1, i, j)
            if delta < 0:
                moves.append(m)
        return moves

    def if_has_edge(self, cycles, a, b):
        status = None, 0

        for i, c in enumerate(cycles):
            for j in range(len(c) - 1):
                if int(a) == int(c[j]) and int(b) == int(c[j+1]):
                    status = i, 1
                    break
                if int(a) == int(c[j+1]) and int(b) == int(c[j]):
                    status = i, 2
                    break

            if status != (None, 0):
                break
            if int(a) == int(c[-1]) and int(b) == int(c[0]):
                status = i, 1
                break
            if int(a) == int(c[0]) and int(b) == int(c[-1]):
                status = i, 2
                break

        return status


class SteepestSearch:
    def __init__(self, local_search):
        self.ls = local_search

    def search(self, cycles):
        while True:
            m = self.ls.generate_starting_moves(cycles)
            if not m:
                break
            self.ls.commit(cycles, min(m, key=lambda x: x[1]))
        return cycles


class CandidatesSearch:
    def __init__(self, local_search):
        self.ls = local_search
        self.vertices = self.ls.get_distance_matrix()

    def search(self, cycles=None, t=10):
        closest = np.delete(np.argsort(self.vertices)[:, :t+1], 0, 1)

        while True:
            best_delta, best_m = 0, None

            for i in range(len(self.vertices)):
                for j in closest[i]:
                    c1, x = self.ls.find_vertex(cycles, i)
                    c2, y = self.ls.find_vertex(cycles, j)
                    delta, m = None, None

                    if c1 == c2:
                        c = cycles[c1]
                        i1, j1 = i, c[(x+1) % len(c)]
                        i2, j2 = j, c[(y+1) % len(c)]

                        delta = self.ls.get_delta_of_swamping_edge(
                            i1, j1, i2, j2)
                        m = 'swaping_edge', delta, i1, j1, i2, j2
                    else:
                        delta, m = self.ls.get_delta_and_move_of_swamping_vertex(
                            cycles, c1, c2, x, y)

                    if delta < best_delta:
                        best_delta = delta
                        best_m = m

            if best_m is None:
                break

            self.ls.commit(cycles, best_m)
        return cycles


class LM:
    def __init__(self, local_search):
        self.ls = local_search

    def get_new_moves(self, cycles, m):
        moves = []
        if m[0] == 'swaping_edge':
            c, _ = self.ls.find_vertex(cycles, m[2])
            cycle = cycles[c]

            for i, j in self.ls.generate_edges(len(cycle)):
                a, b, c, d = cycle[i], cycle[(
                    i+1) % len(cycle)], cycle[j], cycle[(j+1) % len(cycle)]
                delta = self.ls.get_delta_of_swamping_edge(a, b, c, d)
                if delta < 0:
                    moves.append(('swaping_edge', delta, a, b, c, d))

        elif m[0] == 'swaping_vertex':
            i = cycles[m[2]].index(m[8])
            j = cycles[m[3]].index(m[5])

            for k in range(len(cycles[m[3]])):
                delta, m = self.ls.get_delta_and_move_of_swamping_vertex(
                    cycles, m[2], m[3], i, k)
                if delta < 0:
                    moves.append(m)
            for k in range(len(cycles[m[2]])):
                delta, m = self.ls.get_delta_and_move_of_swamping_vertex(
                    cycles, m[3], m[2], j, k)
                if delta < 0:
                    moves.append(m)
        return moves

    def search(self, cycles):
        moves = sorted(self.ls.generate_starting_moves(
            cycles), key=lambda x: x[1])

        while True:
            moves_to_delete = []
            best_m = None

            for idx, m in enumerate(moves):
                if m[0] == 'swaping_edge':
                    c1, s1 = self.ls.if_has_edge(cycles, m[2], m[3])
                    c2, s2 = self.ls.if_has_edge(cycles, m[4], m[5])

                    if c1 != c2 or s1 == 0 or s2 == 0:
                        moves_to_delete.append(idx)
                    elif s1 == s2 == 1:
                        best_m = m
                        moves_to_delete.append(idx)
                        break
                    elif s1 == s2 == 2:
                        best_m = 'swaping_edge', m[1], m[3], m[2], m[5], m[4]
                        moves_to_delete.append(idx)
                        break

                elif m[0] == 'swaping_vertex':
                    _, s1 = self.ls.if_has_edge(cycles, m[4], m[5])
                    _, s2 = self.ls.if_has_edge(cycles, m[5], m[6])
                    _, s3 = self.ls.if_has_edge(cycles, m[7], m[8])
                    _, s4 = self.ls.if_has_edge(cycles, m[8], m[9])

                    if m[2] == m[3] or not all([s1, s2, s3, s4]):
                        moves_to_delete.append(idx)
                    elif s1 == s2 and s3 == s4:
                        moves_to_delete.append(idx)
                        best_m = m
                        break

            if best_m is None:
                break

            for i in reversed(moves_to_delete):
                del(moves[i])
            self.ls.commit(cycles, best_m)

            moves = sorted(list(set(moves).union(
                set(self.get_new_moves(cycles, best_m)))), key=lambda x: x[1])

        return cycles


if __name__ == '__main__':
    paths = ["C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB3/kroA200.txt",
             "C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB3/kroB200.txt"]

    for i in paths:
        ls = LocalSearch()
        ls.read_data(i)

        cs = CandidatesSearch(ls)
        ss = SteepestSearch(ls)
        lm = LM(ls)

        for j in [cs, ss, lm]:
            start = time.time()
            a_cycles = np.array([])
            scores = np.array([])

            min_score = None
            best_cycles = None

            for _ in tqdm(range(100)):
                cycles = ls.get_random_cycles()
                cycles = j.search(cycles)
                scores = np.append(scores, ls.get_scores(cycles))
                if min_score == None or ls.get_scores(cycles) < min_score:
                    best_cycles = cycles

            print("TIME:", time.time() - start)
            print("MIN | MEAN | MAX")
            print(np.min(scores), np.mean(scores), np.max(scores))

            plt.subplots()
            ls.make_visualizations(best_cycles[0], color='blue')
            ls.make_visualizations(best_cycles[1], color='yellow')
            plt.scatter(ls.coordinate_matrix[:, 1],
                        ls.coordinate_matrix[:, 2], color='black')
            plt.show()
