import sys
import numpy as np
import pandas as pd

def load(fname):
    f = open(fname, 'r').readlines()
    n = len(f)
    ret = {}
    for l in f:
        l = l.split('\n')[0].split(',')
        i = l[0]
        ret[i] = {}
        for j in range(n):
            if str(j) in l[1:]:
                ret[i][str(j)] = 1
            else:
                ret[i][str(j)] = 0
    ret = pd.DataFrame(ret).values
    return ret
def get_tran(g):
    # TODO
    n = len(g)
    divide = []
    result = np.zeros((n,n))
    for i in range(n):
        summation = 0
        for j in range(n):
            summation += g[j][i]
        divide.append(summation)
    for i in range(n):
        for j in range(n):
            result[i][j] = float(g[i][j])/divide[j]
    return result

def cal_rank(t, d = 0.85, max_iterations = 1000, alpha = 0.001):
    # TODO
    N = len(t)
    R0 = []
    initial = []
    for i in range(N):
        R0.append([(1-d)/N])
        initial.append([(1/N)])
    for i in range(max_iterations):
        not_changed = initial
        initial = np.add(R0,d*np.matmul(t,initial))
        error = dist(initial,not_changed)
        if error <= alpha:
            return initial





def save(t, r):
    # TODO
    np.savetxt("1.txt",t)
    np.savetxt("2.txt",r)

def dist(a, b):
    return np.sum(np.abs(a-b))

def main():
    graph = load(sys.argv[1])
    print ("graph = ",graph)
    transition_matrix = get_tran(graph)
    rank = cal_rank(transition_matrix)
    save(transition_matrix, rank)
    print ("Transition = ",transition_matrix)
    print ("Rank = ",rank)

if __name__ == '__main__':
    main()

