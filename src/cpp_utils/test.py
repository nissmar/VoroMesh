import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("./build/")
from VoroMeshUtils import compute_voromesh

print("ok")
points = np.random.rand(10000, 3)
values = np.random.rand(10000)-.5

vertices, faces = (compute_voromesh(points, values))

print(vertices, faces)