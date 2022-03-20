from doctest import script_from_examples
import numpy as np
import math
import random as ran
from tqdm import tqdm
import matplotlib.pyplot as plt


class TSP_Solvers:
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

    def nearest_neighbor(self, v1, random=False):
        vertexes = list(range(self.distance_matrix.shape[0]))
        vertexes.remove(v1)

        if not random:
            v2 = np.argmax(self.distance_matrix[v1, :])
        else:
            v2 = ran.choice(vertexes)

        vertexes.remove(v2)
        cycles = [[v1], [v2]]

        while vertexes:
            for c in cycles:
                best_end = vertexes[np.argmin(
                    self.distance_matrix[c[-1], vertexes])]
                end_value = self.distance_matrix[c[-1], best_end]
                best_front = vertexes[np.argmin(
                    self.distance_matrix[c[0], vertexes])]
                front_value = self.distance_matrix[c[0], best_front]
                if front_value < end_value:
                    c.insert(0, best_front)
                    vertexes.remove(best_front)
                else:
                    c.insert(-1, best_end)
                    vertexes.remove(best_end)
        return cycles

    def greedy_cycle(self, v1, random=False):
        vertexes = list(range(self.distance_matrix.shape[0]))
        vertexes.remove(v1)

        if not random:
            v2 = np.argmax(self.distance_matrix[v1, :])
        else:
            v2 = ran.choice(vertexes)

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

    def regret_heuristics(self, v1, random=False):
        vertexes = list(range(self.distance_matrix.shape[0]))
        vertexes.remove(v1)

        if not random:
            v2 = np.argmax(self.distance_matrix[v1, :])
        else:
            v2 = ran.choice(vertexes)

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

    def get_scores(self, matrix):
        matrix += [matrix[0]]
        s1 = sum(self.distance_matrix[matrix[0][i], matrix[0][i+1]]
                 for i in range(len(matrix[0]) - 1))
        s2 = sum(self.distance_matrix[matrix[1][i], matrix[1][i+1]]
                 for i in range(len(matrix[1]) - 1))
        return s1 + s2

    def make_visualizations(self, a, color):
        a += [a[0]]
        for i in range(len(a) - 1):
            plt.plot([self.coordinate_matrix[a[i]][1], self.coordinate_matrix[a[i+1]][1]],
                     [self.coordinate_matrix[a[i]][2], self.coordinate_matrix[a[i+1]][2]], color=color)


tsp = TSP_Solvers()
paths = ["C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB1/kroA100.txt",
         "C:/Users/Maksim/Desktop/repos/PUT-WI-SI/IMO/LAB1/kroB100.txt"]

for i in paths:
    nn_array = np.array([])
    gc_array = np.array([])
    rh_array = np.array([])
    tsp.read_data(i)

    for j in tqdm(range(0, 100)):
        nn_array = np.append(nn_array, tsp.get_scores(tsp.nearest_neighbor(j)))
        gc_array = np.append(gc_array, tsp.get_scores(tsp.greedy_cycle(j)))
        rh_array = np.append(
            rh_array, tsp.get_scores(tsp.regret_heuristics(j)))

    print("MIN | MEAN | MAX")
    print(np.min(nn_array), np.mean(nn_array), np.max(nn_array))
    print(np.min(gc_array), np.mean(gc_array), np.max(gc_array))
    print(np.min(rh_array), np.mean(rh_array), np.max(rh_array))

    best_paths = [tsp.nearest_neighbor(np.argmin(nn_array)),
                  tsp.greedy_cycle(np.argmin(gc_array)),
                  tsp.regret_heuristics(np.argmin(rh_array))]

    for j in best_paths:
        plt.subplots()
        tsp.make_visualizations(j[0], color='blue')
        tsp.make_visualizations(j[1], color='yellow')
        plt.scatter(tsp.coordinate_matrix[:, 1],
                    tsp.coordinate_matrix[:, 2], color='black')
        plt.show()
