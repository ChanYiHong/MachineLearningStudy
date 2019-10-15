#Ex1-d
import numpy as np
n = int(input("Enter matrix size (n): "))
M = np.random.random((n,n))
print("M =\n", M)
M1 = np.linalg.inv(M)
print("M**(-1) =\n", M1)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=12)
print("M x M**(-1) =\n", M @ M1 + 1e-13)
