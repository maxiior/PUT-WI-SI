from threading import local
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

    def regret_heuristics(self, cycles, vertexes):
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

                if vertexes == []:
                    break

                if len(scores_matrix[0]) == 1:
                    best_vertex = vertexes[np.argmin(scores_matrix)]
                    c.insert(0, best_vertex)
                else:
                    sorted_array = np.sort(scores_matrix)[:, :2]
                    if sorted_array != []:
                        regret = np.array([i[1]-i[0] for i in sorted_array])
                        idx = np.argmax(regret - np.min(scores_matrix, axis=1))
                        best_vertex = vertexes[idx]
                        c.insert(np.argmin(scores_matrix[idx]), best_vertex)
                    else:
                        continue

                if best_vertex != None:
                    vertexes.remove(best_vertex)
        return cycles

    def get_distance_matrix(self):
        return self.distance_matrix

    def get_scores(self, matrix):
        matrix = deepcopy(matrix)
        matrix[0] += [matrix[0][0]]
        matrix[1] += [matrix[1][0]]
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

    def get_delta_and_move_of_swamping_vertex(self, cycles, ic1, ic2, i, j):
        c1, c2 = cycles[ic1], cycles[ic2]
        x1, y1, z1 = c1[(i-1) % len(c1)], c1[i], c1[(i+1) % len(c1)]
        x2, y2, z2 = c2[(j-1) % len(c2)], c2[j], c2[(j+1) % len(c2)]
        d = self.distance_matrix
        delta = d[x1, y2] + d[z1, y2] + d[x2, y1] + d[z2, y1] - \
            d[x1, y1] - d[z1, y1] - d[x2, y2] - d[z2, y2]
        move = 'swaping_vertex', delta, ic1, ic2, x1, y1, z1, x2, y2, z2
        return delta, move

    def generate_vertices(self, n, m):
        return [(i, j) for i in range(n) for j in range(m)]

    def get_delta_of_swamping_edge(self, a, b, c, d):
        return self.distance_matrix[a, c] + self.distance_matrix[b, d] - self.distance_matrix[a, b] - self.distance_matrix[c, d]

    def generate_edges(self, n):
        return [(j, (j+i) % n) for j in range(n) for i in range(2, n-1)]

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

    def find_vertex(self, c, i):
        try:
            return 0, c[0].index(i)
        except:
            try:
                return 1, c[1].index(i)
            except:
                return None, None

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

    def generate_vertex_move(self, cycles):
        c = self.generate_vertices(len(cycles[0]), len(cycles[1]))
        random.shuffle(c)
        return self.get_delta_and_move_of_swamping_vertex(cycles, 0, 1, c[0][0], c[0][1])[1]

    def generate_edge_move(self, cycles):
        k = random.sample(range(2), 2)
        cycle = cycles[k[0]]
        candidates = self.generate_edges(len(cycle))
        random.shuffle(candidates)
        return 'swaping_edge', 0, cycle[candidates[0][0]], cycle[(candidates[0][0]+1) % len(cycle)], cycle[candidates[0][1]], cycle[(candidates[0][1]+1) % len(cycle)]

    def generate_move(self, cycles):
        moves = [self.generate_vertex_move, self.generate_edge_move]
        r = random.sample(range(2), 2)
        m = moves[r[0]](cycles)
        return moves[r[1]](cycles) if m is None else m


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


class EvolutionSearch:
    def __init__(self, local_search):
        self.ls = local_search

    def combine(self, sol1, sol2):
        sol1, sol2 = deepcopy(sol1), deepcopy(sol2)
        
        remaining = []
        for cyc1 in sol1:
            n = len(cyc1)
            if n == 1:
                continue
            for i in range(n):
                p, q = cyc1[i], cyc1[(i+1)%n]
                if p == -1 or q == -1 or p == q:
                    continue
                found = False
                for cyc2 in sol2:
                    m = len(cyc2)
                    for j in range(m):
                        u, v = cyc2[j], cyc2[(j+1)%m]
                        if (p == u and q == v) or (p == v and q == u):
                            found = True
                            break
                    if found:
                        break
                        
                if not found:
                    remaining.append(cyc1[i])
                    remaining.append(cyc1[(i+1)%n])
                    cyc1[i] = -1
                    cyc1[(i+1)%n] = -1
                    
            for i in range(1, n):
                x, y, z = cyc1[(i-1)%n], cyc1[i], cyc1[(i+1)%n]
                if x == z == -1 and y != -1:
                    remaining.append(y)
                    cyc1[i] = -1
                    
            for i in range(1, n):
                x = cyc1[i]
                if x != -1 and np.random.rand() < 0.2:
                    remaining.append(x)
                    cyc1[i] = -1
                    
        a = [x for x in sol1[0] if x != -1]
        b = [x for x in sol1[1] if x != -1]
        return self.ls.regret_heuristics([a, b], remaining)

    def search(self, population_size=3, limit=10, score_threshold=40, iterations_threshold=300, local=True):
        ss = SteepestSearch(self.ls)
        populations = np.array([(i, self.ls.get_scores(i)) for i in [ss.search(self.ls.get_random_cycles()) for _ in tqdm(range(population_size))]])

        start = time.time()
        best_scores = []
        worst_scores = []

        i, update_i, best_idx = 0, 0, 0
        current_best = populations[0][1]

        while limit > time.time() - start:
            i += 1
            indexes = np.arange(population_size)
            np.random.shuffle(indexes)

            worst_i = np.argmax(populations[:,1])
            worst_score = populations[worst_i][1]

            result = self.combine(populations[indexes[0]][0], populations[indexes[1]][0])

            print(np.array(result))

            # if local:
            #     sol = ss.search(sol, inplace=True)
            # result_score = self.ls.get_scores(np.array([result[0], result[1]]))

            # if result_score < current_best:
            #     populations[best_idx] = result, result_score
            # elif result_score < worst_score and not any(abs(result_score - j) < score_threshold for j in populations[:,1]):
            #     populations[worst_i] = result, result_score

            # best_result, best_score = populations[np.argmin(populations[:,1])]
            # best_scores.append(best_score)
            # worst_scores.append(worst_score)

            # if best_score < current_best:
            #     current_best = best_score
            #     update_i = i

            # if i - update_i > iterations_threshold:
            #     break

        #     if plot:
        #         clear_output(wait=True)
        #         plt.figure()
        #         plot_solution(coords, best_sol)
        #         plt.title(f'iteration {i} (best: {best_score}, worst: {worst_score})')
        #         plt.show()
        #         plt.figure()
        #         plt.plot(best_scores, label='best score')
        #         plt.plot(worst_scores, label='worst score')
        #         plt.legend()
        #         plt.show()

        return best_result


if __name__ == '__main__':
    paths = ["C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB5/kroA200.txt",
             "C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB5/kroB200.txt"]

    ls = LocalSearch()
    ls.read_data(paths[0])

    es = EvolutionSearch(ls)
    es.search()

    # for path in paths:
    #     ls = LocalSearch()
    #     ls.read_data(path)
    #     ss = SteepestSearch(ls)

    #     msls_scores = []
    #     ils1_scores = []
    #     ils2_scores = []

    #     msls_times = []

    #     msls_cycles = []
    #     ils1_cycles = []
    #     ils2_cycles = []

    #     for i, _ in enumerate(range(10)):
    #         start = time.time()
    #         msls = MSLS(ls, ss)
    #         cycle = msls.search()
    #         msls_scores.append(ls.get_scores(cycle))
    #         msls_cycles.append(cycle)
    #         msls_times.append(time.time()-start)
    #         print("MSLS: {0}".format(i))

    #     print("DONE")

    #     for i, _ in enumerate(range(10)):
    #         ils1 = ILS1(ls)
    #         ils = ILS(ls, ils1, ss)
    #         cycle = ils.search(np.mean(msls_times))
    #         ils1_scores.append(ls.get_scores(cycle))
    #         ils1_cycles.append(cycle)
    #         print("ILS1: {0}".format(i))

    #     print("DONE")

    #     for i, _ in enumerate(range(10)):
    #         ils2 = ILS2(ls)
    #         ils = ILS(ls, ils2, ss)
    #         cycle = ils.search(np.mean(msls_times))
    #         ils2_scores.append(ls.get_scores(cycle))
    #         ils2_cycles.append(cycle)
    #         print("ILS2: {0}".format(i))

    #     print("DONE")

    #     print("TIMES")
    #     print("MIN | MEAN | MAX")
    #     print(np.min(msls_times), np.mean(msls_times), np.max(msls_times))
    #     print("SCORES")
    #     print("MIN | MEAN | MAX")
    #     print("MSLS")
    #     print(np.min(msls_scores), np.mean(msls_scores), np.max(msls_scores))
    #     print("ILS1")
    #     print(np.min(ils1_scores), np.mean(ils1_scores), np.max(ils1_scores))
    #     print("ILS2")
    #     print(np.min(ils2_scores), np.mean(ils2_scores), np.max(ils2_scores))

    #     plt.subplots()
    #     ls.make_visualizations(
    #         msls_cycles[np.argmin(msls_scores)][0], color='blue')
    #     ls.make_visualizations(
    #         msls_cycles[np.argmin(msls_scores)][1], color='yellow')
    #     plt.scatter(ls.coordinate_matrix[:, 1],
    #                 ls.coordinate_matrix[:, 2], color='black')
    #     plt.show()

    #     plt.subplots()
    #     ls.make_visualizations(
    #         ils1_cycles[np.argmin(ils1_scores)][0], color='blue')
    #     ls.make_visualizations(
    #         ils1_cycles[np.argmin(ils1_scores)][1], color='yellow')
    #     plt.scatter(ls.coordinate_matrix[:, 1],
    #                 ls.coordinate_matrix[:, 2], color='black')
    #     plt.show()

    #     plt.subplots()
    #     ls.make_visualizations(
    #         ils2_cycles[np.argmin(ils2_scores)][0], color='blue')
    #     ls.make_visualizations(
    #         ils2_cycles[np.argmin(ils2_scores)][1], color='yellow')
    #     plt.scatter(ls.coordinate_matrix[:, 1],
    #                 ls.coordinate_matrix[:, 2], color='black')
    #     plt.show()
