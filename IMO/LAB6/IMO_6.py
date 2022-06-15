import numpy as np
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import matplotlib.pyplot as plt
import time
from scipy import stats


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

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_scores(self, matrix):
        m = deepcopy(matrix)
        a = np.append(m[0], m[0][0])
        b = np.append(m[1], m[1][0])
        s1 = sum(self.distance_matrix[a[i], a[i+1]]
                 for i in range(len(a) - 1))
        s2 = sum(self.distance_matrix[b[i], b[i+1]]
                 for i in range(len(b) - 1))
        return s1 + s2

    def get_random_cycles(self):
        n = self.distance_matrix.shape[0]
        vertexes = list(range(n))
        random.shuffle(vertexes)
        return [vertexes[:n//2], vertexes[n//2:]]

    def greedy_cycle(self, v1, random=False):
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

                best_vertex = best_value = insert_position = None

                for idx, i in enumerate(scores_matrix):
                    cur_value = np.min(i)
                    if best_vertex == None or best_value > cur_value:
                        best_value = cur_value
                        best_vertex = vertexes[idx]
                        insert_position = np.argmin(i)

                c.insert(insert_position, best_vertex)
                vertexes.remove(best_vertex)
        return cycles

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

    def compare_cycles_vertexes(self, c1, c2):
        a, b, c, d = 0, 0, 0, 0

        for i in c1[0]:
            if i in c2[0]:
                a += 1

        for i in c1[1]:
            if i in c2[1]:
                b += 1

        for i in c1[0]:
            if i in c2[1]:
                c += 1

        for i in c1[1]:
            if i in c2[0]:
                d += 1

        return np.max([(a+b)/(2*len(c1[0])), (c+d)/(2*len(c1[0]))])

    def compare_cycles_edges(self, c1, c2):
        edges11 = np.array([(c1[0][idx], c1[0][idx+1]) if idx < len(c1[0])-1
                            else (c1[0][-1], c1[0][0]) for idx, _ in enumerate(c1[0])])
        edges12 = np.array([(c1[1][idx], c1[1][idx+1]) if idx < len(c1[1])-1
                            else (c1[1][-1], c1[1][0]) for idx, _ in enumerate(c1[1])])

        edges21 = np.array([(c2[0][idx], c2[0][idx+1]) if idx < len(c2[0])-1
                            else (c2[0][-1], c2[0][0]) for idx, _ in enumerate(c2[0])])
        edges22 = np.array([(c2[1][idx], c2[1][idx+1]) if idx < len(c2[1])-1
                            else (c2[1][-1], c2[1][0]) for idx, _ in enumerate(c2[1])])

        a, b, c, d = 0, 0, 0, 0

        for i in edges11:
            if i in edges21:
                a += 1

        for i in edges12:
            if i in edges22:
                b += 1

        for i in edges11:
            if i in edges22:
                c += 1

        for i in edges12:
            if i in edges21:
                d += 1

        return np.max([(a+b)/(2*len(c1[0])), (c+d)/(2*len(c1[0]))])

    def avg_edges_similarity(self, c, cycles):
        a = np.array([self.compare_cycles_edges(c, i)
                     for i in cycles])
        return np.mean(a)

    def avg_vertexes_similarity(self, c, cycles):
        a = np.array([self.compare_cycles_vertexes(c, i)
                     for i in cycles])
        return np.mean(a)


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
        while True:
            move = random.choice([0, 1])
            if self.moves[move](cycles) >= 0:
                if self.moves[1-move](cycles) >= 0:
                    break
        return cycles


if __name__ == '__main__':
    paths = ["C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB6/kroA100.txt",
             "C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB6/kroB100.txt"]

    ls = LocalSearch()
    ls.read_data(paths[1])

    gs = GreedySearch(True, ls)

    cycles = np.array([gs.greedy_search(ls.get_random_cycles())
                       for _ in tqdm(range(1000))])

    scores = np.array([ls.get_scores(i) for i in cycles])
    best_cycles = cycles[np.argmin(scores)]

    touples = np.array([(scores[idx], ls.compare_cycles_edges(
        best_cycles, i)) for idx, i in enumerate(cycles)])
    touples = np.array([i for i in touples if 1 not in i]).T
    print(stats.pearsonr(touples[0], touples[1]))

    touples = np.array([(scores[idx], ls.compare_cycles_vertexes(
        best_cycles, i)) for idx, i in enumerate(cycles)])
    touples = np.array([i for i in touples if 1 not in i]).T
    print(stats.pearsonr(touples[0], touples[1]))

    touples = np.array([(scores[idx], ls.avg_edges_similarity(
        i, cycles)) for idx, i in enumerate(cycles)])
    touples = np.array([i for i in touples if 1 not in i]).T
    print(stats.pearsonr(touples[0], touples[1]))

    touples = np.array([(scores[idx], ls.avg_vertexes_similarity(
        i, cycles)) for idx, i in enumerate(cycles)])
    touples = np.array([i for i in touples if 1 not in i]).T
    print(stats.pearsonr(touples[0], touples[1]))
