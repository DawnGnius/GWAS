import numpy as np
X = np.random.random((3,2))
X_train = np.random.random((5,2))
print('X:')
print(X)
print('X_train:')
print(X_train)

#way 3:use no loops
dist = np.reshape(np.sum(X**2,axis=1),(X.shape[0],1))+ np.sum(X_train**2,axis=1)-2*X.dot(X_train.T)
print('--------------------')
print('way 3 result:')
print(dist)
