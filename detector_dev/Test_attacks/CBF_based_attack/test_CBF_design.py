import numpy as np
import matplotlib.pyplot as plt

from hull_classification_utils import *

from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay





if __name__ == '__main__':
	points = np.array([[1,0],[-1,0],[0,1],[0,-1],[0,0]])

	unit_hull = ConvexHull(points)

	A = unit_hull.equations[:,:-1]
	b = unit_hull.equations[:,-1]

	A_cbf


