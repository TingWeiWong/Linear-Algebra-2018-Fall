import numpy as np
def get_tran(g):
	# TODO
	n = len(g)
	for i in range(n):
		summation = 0
		for j in range(n):
			summation += g[j][i]
		for j in range(n):
			if summation !=0:
				g[j][i] = float(g[j][i]) / float(summation)
	result = g
	return result	
test1 = [[0,0,0,0,1],
		 [1,0,1,1,0],
		 [1,0,0,0,1],
		 [0,0,0,0,0],
		 [0,1,0,0,0]]
for i in range(5):
	for j in range(5):
		test1[i][j] /= 2
print ("After for loop = ",test1)
test2 = np.transpose(test1)
print (test1)
print (get_tran(test1))

x = [[1],
	 [2]]
y = [[3],
	 [4]]
print (np.add(x,y))
print ((5-2)/2)

