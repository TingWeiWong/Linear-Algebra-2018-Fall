import numpy as np
def get_tran(g):
	# TODO
	n = len(g)
	for i in range(n):
		summation = 0
		for j in range(n):
			summation += g[i][j]
		for j in range(n):
			if summation != 0 :
				g[i][j] = float(g[i][j]) / float(summation)
	result = g
	return result
test1 = [[0,0,0,0,1],
		 [1,0,1,1,0],
		 [1,0,0,0,1],
		 [0,0,0,0,0],
		 [0,1,0,0,0]]
test1 = np.transpose(test1)
print (test1)
print (get_tran(test1))
