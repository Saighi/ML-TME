# coding: utf-8
import numpy as np


class Histogramme:



    def learn(self, geo_mat, ymin, ymax, xmin, xmax, steps):
        hauteur = (ymax - ymin) / steps
        largeur = (xmax - xmin) / steps
        hist = np.zeros((steps, steps))
        n = len(geo_mat)

        for i in range(0, steps):
            print(i * hauteur, i+1 * hauteur)
            for j in range(0, steps):
                c = 0
                for k in geo_mat:

                    if i * hauteur < k[0]-ymin < (i+1) * hauteur and j  * largeur < k[1]-xmin < (j+1) * largeur:
                        c += 1

                hist[i, j] = c / (n*(largeur*hauteur))
        return hist

# coding: utf-8
