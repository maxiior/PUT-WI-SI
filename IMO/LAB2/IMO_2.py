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

    def get_random_cycles(self):
        n = self.distance_matrix.shape[0]
        vertexes = list(range(n))
        random.shuffle(vertexes)
        return [vertexes[:n//2], vertexes[n//2:]]

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_scores(self, matrix):
        matrix += [matrix[0]]
        s1 = sum(self.distance_matrix[matrix[0][i], matrix[0][i+1]]
                 for i in range(len(matrix[0]) - 1))
        s2 = sum(self.distance_matrix[matrix[1][i], matrix[1][i+1]]
                 for i in range(len(matrix[1]) - 1))
        return s1 + s2

    def get_inside_candidates(self, cycle):
        c = []
        l = len(cycle)
        for i in range(l):
            for j in range(i+1, l):
                c.append([i, j])
        return c

    def replace_vertices_inside(self, cycle, i, j):
        tmp = cycle[i]
        cycle[i] = cycle[j]
        cycle[j] = tmp

    def replace_vertices_outside(self, cycles, i, j):
        tmp = cycles[0][i]
        cycles[0][i] = cycles[1][j]
        cycles[1][j] = tmp

    def get_outside_candidates(self, cycles):
        return [(i, j) for j in range(len(cycles[1])) for i in range(len(cycles[0]))]

    def replace_edges_inside(self, cycle, i, j):
        if (i, j) == (0, len(cycle)-1):
            temp = cycle[i]
            cycle[i] = cycle[j]
            cycle[j] = temp
        cycle[i:j+1] = reversed(cycle[i:j+1])

    def replace_vertices_inside_delta_score(self, cycle, i, j):
        matrix = self.distance_matrix
        l = len(cycle)

        a, b, c = cycle[(i - 1) % l], cycle[i], cycle[(i+1) % l]
        d, e, f = cycle[(j-1) % l], cycle[j], cycle[(j+1) % l]

        if j-i == 1:
            return matrix[a, e] + matrix[b, f] - matrix[a, b] - matrix[e, f]
        elif (i, j) == (0, l-1):
            return matrix[e, c] + matrix[d, b] - matrix[b, c] - matrix[d, e]
        else:
            return matrix[a, e] + matrix[e, c] + matrix[d, b] + matrix[b, f] - matrix[a, b] - matrix[b, c] - matrix[d, e] - matrix[e, f]

    def replace_edges_inside_delta(self, cycle, i, j):
        matrix = self.distance_matrix
        l = len(cycle)

        if (i, j) == (0, len(cycle)-1):
            a, b, c, d = cycle[i], cycle[(
                i+1) % l], cycle[(j-1) % l], cycle[j]
        else:
            a, b, c, d = cycle[(i - 1) %
                               l], cycle[i], cycle[j], cycle[(j+1) % l]
        return matrix[a, c] + matrix[b, d] - matrix[a, b] - matrix[c, d]

    def replace_vertices_outside_delta_score(self, cycles, i, j):
        score = 0
        matrix = self.distance_matrix
        for x, y in enumerate([i, j]):
            l = len(cycles[x])
            a, b, c = cycles[x][(y - 1) %
                                l], cycles[x][y], cycles[x][(y+1) % l]
            v2 = cycles[1-x][i if y == j else j]
            score += matrix[a, v2] + matrix[v2, c] - \
                matrix[a, b] - matrix[b, c]
        return score


class RandomSearch:
    def __init__(self, local_search):
        self.ls = local_search
        self.moves = [self.replace_vertices_outside,
                      self.replace_vertices_inside, self.replace_edges_inside]

    def replace_vertices_outside(self, cycles):
        i, j = random.choice(self.ls.get_outside_candidates(cycles))
        self.ls.replace_vertices_outside(cycles, i, j)

    def replace_vertices_inside(self, cycles):
        cycle = random.choice([0, 1])
        i, j = random.choice(self.ls.get_inside_candidates(cycles[cycle]))
        self.ls.replace_vertices_inside(cycles[cycle], i, j)

    def replace_edges_inside(self, cycles):
        cycle = random.choice([0, 1])
        i, j = random.choice(self.ls.get_inside_candidates(cycles[cycle]))
        self.ls.replace_edges_inside(cycles[cycle], i, j)

    def random_search(self, cycles, limit):
        best_cycles = cycles
        score = self.ls.get_scores(deepcopy(cycles))
        start = time.time()

        while time.time()-start < limit:
            random.choice(self.moves)(cycles)
            new = self.ls.get_scores(deepcopy(cycles))
            if new < score:
                best_cycles = cycles
                score = new
        return best_cycles


class GreedySearch:
    def __init__(self, mode, local_search):
        self.ls = local_search
        self.moves = [self.replace_outside, self.replace_inside]

        if mode == True:
            self.f = self.ls.replace_vertices_inside_delta_score
            self.g = self.ls.replace_vertices_inside
        else:
            self.f = self.ls.replace_edges_inside_delta
            self.g = self.ls.replace_edges_inside

    def replace_outside(self, cycles):
        c = self.ls.get_outside_candidates(cycles)
        random.shuffle(c)
        for i, j in c:
            score = self.ls.replace_vertices_outside_delta_score(cycles, i, j)
            if score < 0:
                self.ls.replace_vertices_outside(cycles, i, j)
                return score
        return score

    def replace_inside(self, cycles):
        for cycle in random.sample(range(2), 2):
            c = self.ls.get_inside_candidates(cycles[cycle])
            random.shuffle(c)
            for i, j in c:
                score = self.f(cycles[cycle], i, j)
                if score < 0:
                    self.g(cycles[cycle], i, j)
                    return score
        return score

    def greedy_search(self, cycles):
        start = time.time()
        while True:
            move = random.choice([0, 1])
            if self.moves[move](cycles) >= 0:
                if self.moves[1-move](cycles) >= 0:
                    break
        return time.time()-start, cycles


class SteepestSearch:
    def __init__(self, mode, local_search):
        self.ls = local_search
        self.moves = [self.replace_outside, self.replace_inside]

        if mode == True:
            self.f = self.ls.replace_vertices_inside_delta_score
            self.g = self.ls.replace_vertices_inside
        else:
            self.f = self.ls.replace_edges_inside_delta
            self.g = self.ls.replace_edges_inside

    def replace_outside(self, cycles):
        c = self.ls.get_outside_candidates(cycles)
        scores = np.array(
            [self.ls.replace_vertices_outside_delta_score(cycles, i, j) for i, j in c])
        min_idx = np.argmin(scores)

        if scores[min_idx] < 0:
            return self.ls.replace_vertices_outside, cycles, c[min_idx], scores[min_idx]
        return None, None, None, scores[min_idx]

    def replace_inside(self, cycles):
        c = self.ls.get_inside_candidates(
            cycles[0]), self.ls.get_inside_candidates(cycles[1])
        scores = np.array([[self.f(cycles[idx], i, j)
                          for i, j in c[idx]] for idx in range(len(c))])
        i, j = np.unravel_index(np.argmin(scores), scores.shape)

        score = scores[i, j]
        if score < 0:
            return self.g, cycles[i], c[i][j], score
        return None, None, None, score

    def steepest_search(self, cycles):
        cycles = deepcopy(cycles)
        start = time.time()

        while True:
            funs_a = []
            cycles_a = []
            c_a = []
            scores_a = []
            for i in self.moves:
                fun, cy, c, scores = i(cycles)
                funs_a.append(fun)
                cycles_a.append(cy)
                c_a.append(c)
                scores_a.append(scores)

            idx = np.argmin(scores_a)

            if scores_a[idx] < 0:
                funs_a[idx](cycles_a[idx], c_a[idx][0], c_a[idx][1])
            else:
                break
        return time.time()-start, cycles


if __name__ == '__main__':
    ls = LocalSearch()
    paths = ["C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB1/kroA100.txt",
             "C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB1/kroB100.txt"]

    ls.read_data(paths[0])

    gs = GreedySearch(True, ls)
    rs = RandomSearch(ls)
    ss = SteepestSearch(True, ls)

    print("RANDOM --- GreedySearch (V) | RandomSearch | SteepestSearch (V)")

    a_gs = []
    a_rs = []
    a_ss = []

    start = time.time()
    for _ in tqdm(range(10)):
        t, gs_cycles = gs.greedy_search(ls.get_random_cycles())
        rs_cycles = rs.random_search(ls.get_random_cycles(), t)
        t, ss_cycles = ss.steepest_search(ls.get_random_cycles())

        a_gs.append(ls.get_scores(gs_cycles))
        a_rs.append(ls.get_scores(rs_cycles))
        a_ss.append(ls.get_scores(ss_cycles))
    print(np.min(a_gs), np.mean(a_gs), np.max(a_gs))
    print(np.min(a_rs), np.mean(a_rs), np.max(a_rs))
    print(np.min(a_ss), np.mean(a_ss), np.max(a_ss))

    print(time.time() - start)

    for i in [gs_cycles, rs_cycles, ss_cycles]:
        plt.subplots()
        ls.make_visualizations(i[0], color='blue')
        ls.make_visualizations(i[1], color='yellow')
        plt.scatter(ls.coordinate_matrix[:, 1],
                    ls.coordinate_matrix[:, 2], color='black')
        plt.show()

    print("REGRET --- GreedySearch (V) | RandomSearch | SteepestSearch (V)")

    a_gs = []
    a_rs = []
    a_ss = []
    start = time.time()
    for i in tqdm(range(10)):
        t, gs_cycles = gs.greedy_search(ls.regret_heuristics(5))
        rs_cycles = rs.random_search(ls.regret_heuristics(5), t)
        t, ss_cycles = ss.steepest_search(ls.regret_heuristics(5))

        a_gs.append(ls.get_scores(gs_cycles))
        a_rs.append(ls.get_scores(rs_cycles))
        a_ss.append(ls.get_scores(ss_cycles))
    print(np.min(a_gs), np.mean(a_gs), np.max(a_gs))
    print(np.min(a_rs), np.mean(a_rs), np.max(a_rs))
    print(np.min(a_ss), np.mean(a_ss), np.max(a_ss))

    print(time.time() - start)

    for i in [gs_cycles, rs_cycles, ss_cycles]:
        plt.subplots()
        ls.make_visualizations(i[0], color='blue')
        ls.make_visualizations(i[1], color='yellow')
        plt.scatter(ls.coordinate_matrix[:, 1],
                    ls.coordinate_matrix[:, 2], color='black')
        plt.show()

    gs = GreedySearch(False, ls)
    ss = SteepestSearch(False, ls)

    print("RANDOM --- GreedySearch (E) | SteepestSearch (E)")

    a_gs = []
    a_ss = []
    start = time.time()
    for i in tqdm(range(10)):
        t, gs_cycles = gs.greedy_search(ls.get_random_cycles())
        t, ss_cycles = ss.steepest_search(ls.get_random_cycles())

        a_gs.append(ls.get_scores(gs_cycles))
        a_ss.append(ls.get_scores(ss_cycles))
    print(np.min(a_gs), np.mean(a_gs), np.max(a_gs))
    print(np.min(a_ss), np.mean(a_ss), np.max(a_ss))

    print(time.time() - start)

    for i in [gs_cycles, ss_cycles]:
        plt.subplots()
        ls.make_visualizations(i[0], color='blue')
        ls.make_visualizations(i[1], color='yellow')
        plt.scatter(ls.coordinate_matrix[:, 1],
                    ls.coordinate_matrix[:, 2], color='black')
        plt.show()

    print("REGRET --- GreedySearch (E) | SteepestSearch (E)")

    a_gs = []
    a_ss = []

    gs_cycles_a = []
    ss_cycles_a = []
    start = time.time()
    for i in tqdm(range(10)):
        t, gs_cycles = gs.greedy_search(ls.regret_heuristics(i))
        t, ss_cycles = ss.steepest_search(ls.regret_heuristics(i))

        a_gs.append(ls.get_scores(gs_cycles))
        a_ss.append(ls.get_scores(ss_cycles))
        gs_cycles_a.append(gs_cycles)
        ss_cycles_a.append(ss_cycles)

    print(time.time() - start)

    print(np.min(a_gs), np.mean(a_gs), np.max(a_gs))
    print(np.min(a_ss), np.mean(a_ss), np.max(a_ss))

    for i in [gs_cycles_a, ss_cycles_a]:
        plt.subplots()
        ls.make_visualizations(i[np.argmin(a_gs)][0], color='blue')
        ls.make_visualizations(i[np.argmin(a_ss)][1], color='yellow')
        plt.scatter(ls.coordinate_matrix[:, 1],
                    ls.coordinate_matrix[:, 2], color='black')
        plt.show()
