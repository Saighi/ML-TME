# coding: utf-8
import numpy as np
from numpy import linalg as LA

class Parzen:

    def __init__(self, geo_mat, window):
        self.window = window
        self.geo_mat = geo_mat
        self.n = len(geo_mat)

    def predict(self, grid):
        probas = np.zeros(len(grid))
        for i in range(len(grid)):
            probas[i] = self.density_at_point(grid[i])
        return probas

    def density_at_point(self, point):
        k = 0
        for i in self.geo_mat:

            if point[0] - (self.window[0] / 2) < i[1] < point[0] + (self.window[0] / 2) and point[1] - (
                    self.window[1] / 2) < i[0] < point[1] + (self.window[1] / 2):
                k += 1
        return k / (self.window[0] * self.window[1] * self.n)


class Gaussian:

    def __init__(self, geo_mat, sigma):
        self.sigma = sigma
        self.geo_mat = geo_mat
        self.n = len(geo_mat)
        self.sum = 0

    def predict(self, grid):
        probas = np.zeros(len(grid))
        for i in range(len(grid)):
            probas[i] = self.density_at_point(grid[i], self.sigma)
        print(probas/self.sum)
        return probas/self.sum

    def density_at_point(self, point,sigma):
        k = 0
        for y,x in self.geo_mat:
            vec = np.array([x,y])
            k += np.exp(-(LA.norm(point-vec)**2)/ (2 * (sigma ** 2))) 
        self.sum += k
        return k


class TwoDGaussian:

    def __init__(self, geo_mat, window):
        self.window = window
        self.geo_mat = geo_mat
        self.n = len(geo_mat)

    def predict(self, grid):
        probas = np.zeros(len(grid))
        for i in range(len(grid)):
            probas[i] = self.density_at_point(grid[i],0.025)
        return probas

    def density_at_point(self, point,sigma):
        k = 0
        for y,x in self.geo_mat:
            vec = np.array([x,y])
            k += (np.exp(-((point[0]-vec[0])**2))/(2 * sigma ** 2)) * (np.exp(-((point[1] - vec[1]) ** 2)) / (2*sigma **2))

        return k/self.n
