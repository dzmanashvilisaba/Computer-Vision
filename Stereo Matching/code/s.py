import numpy as np
from utils import normalize_points

pts1 = np.array([[631, 299], [823, 449], [523, 479], [411, 370], [121, 174], [643, 178], [417, 307], [34, 390]])
pts2 = np.array([[609, 267], [775, 408], [505, 451], [380, 349], [ 91, 165], [618, 146], [399, 285], [13, 385]])
 
print("GODDAMN", np.mean(pts1, axis=1))
pts1, T1 = normalize_points(pts1, pts1.shape[0])
pts2, T2 = normalize_points(pts2, pts2.shape[0]) 
 
A = np.array([])
for i in range(pts2.shape[0]):
    row = np.outer([pts1[i][0], pts1[i][1], 1], [pts2[i][0], pts2[i][1], 1]).reshape(9)
    print(row)
    A = np.append(A, [row])
A = A.reshape(8,9) 


print("\tA = \n", A)
print("\n\nT:",T1)
ATA = np.matmul(np.transpose(A),A)
eigValues, eigVectors = np.linalg.eig(ATA)

print("\tEigen Values of Matrix A is:\n", eigValues)
print("\tEigen Vectors of Matrix A is:\n", eigVectors)


f = eigVectors[8]
fundamental_matrix = f.reshape(3,3)

###     Enforcing rank 2 for the Fundamental Matrix
u, s, v = np.linalg.svd(fundamental_matrix)
fundamental_matrix = np.matmul(np.matmul(u,np.diag(s)),v)

###     Un-normalize fundamental matrix
print(fundamental_matrix)

fundamental_matrix.resize(9)
print(fundamental_matrix)

prod1 = np.matmul(np.transpose(T2), fundamental_matrix)
print(prod1)

prod2 = np.matmul(prod1, T1)
print(prod2)

print(np.mean(np.sqrt(np.sum(np.power(pts1[0:2,:],2), axis=0)))) 
print(np.mean(np.sqrt(np.sum(np.power(pts2[0:2,:],2), axis=0)))) 
print( np.sqrt(2))
print(pts1)
print(pts2)








