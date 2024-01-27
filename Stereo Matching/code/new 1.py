import numpy as np
import cv2

a = np.array([631, 299, 1])
b = np.array([1.2007537,  -1.2007537, 1])

pts1 = np.array([[631, 299], [823, 449], [523, 479], [411, 370], [121, 174], [643, 178], [417, 307], [34, 390]])
pts2 = np.array([[609, 267], [775, 408], [505, 451], [380, 349], [ 91, 165], [618, 146], [399, 285], [13, 385]])
 
 
mean_1 = np.mean(pts1,axis = 0) #[450.375, 330.75] 
mean_2 = np.mean(pts2,axis = 0 )#[423.75, 307] 

scale = 0.007233456346165079

T_scale = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=object)
#  [0.007233456346165079    0                       0]
#  [0                       0.007233456346165079    0]
#  [0                       0                       1]


T_offset = np.array([[1, 0, -mean_1[0]], [0, 1, -mean_1[1]], [0, 0, 1]], dtype=object)
# [[1       0           -450.375]
# [0        1           -330.75]
# [0        0           1]]


T = np.matmul(T_scale, T_offset)
# [0.007233456346165079     0                       -3.2577679019040975]
# [0                        0.007233456346165079    -2.3924656864941]
# [0                        0                       1]

#print(np.matmul(a.reshape(1,3),T))



A = np.array([])
for i in range(pts2.shape[0]):
    row = np.outer([pts1[i][0], pts1[i][1], 1], [pts2[i][0], pts2[i][1], 1]).reshape(9)
    A = np.append(A, [row])
A = A.reshape(8,9) 


ATA = np.matmul(np.transpose(A),A)
eigValues, eigVectors = np.linalg.eig(ATA)

f = eigVectors[8]
fundamental_matrix = f.reshape(3,3)
    
print(fundamental_matrix) 


print("\n\n -- 1 -- \n\n")
ts = cv2.findHomography(pts1, pts2, 0)
print(ts[0])
#[[ 1.08858200e+00  6.53861317e-02 -4.87619411e+01]
# [-2.60797387e-02  1.11983175e+00 -2.49634478e+01]
# [ 6.97830871e-05  1.49472405e-04  1.00000000e+00]]

print("\n\n -- 2 -- \n\n")
rs = cv2.findFundamentalMat(pts1,pts2,2)
print(rs[0])


print("\n\n -- 3 -- \n\n")
