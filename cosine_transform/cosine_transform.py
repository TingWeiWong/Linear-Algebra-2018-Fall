import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_wave(x, path = './wave.png'):
    plt.gcf().clear()
    plt.plot(x)
    plt.xlabel('n')
    plt.ylabel('xn')
    plt.savefig(path)

def plot_ak(a, path = './freq.png'):
    plt.gcf().clear()

    # Only plot the mag of a
    a = np.abs(a)
    plt.plot(a)
    plt.xlabel('k')
    plt.ylabel('ak')
    plt.savefig(path)

def CosineTrans(x, B):
    # TODO
    # implement cosine transform
    inverse_B = inverseMatrix(B)
    a = np.matmul(inverse_B,x)
    a = np.array(a)
    return a

def InvCosineTrans(a, B):
    # TODO
    # implement inverse cosine transform
    return

def gen_basis(N):
    # TODO
    base = np.zeros((N,N))
    k,N = N,N
    for i in range(k):
    	for j in range(N):
    		if i == 0 :
    			base[i][j] = 1.0/np.sqrt(N)
    		else:
    			base[i][j] = np.sqrt(2.0/N) * np.cos((j+0.5)*i*np.pi/N)
    # Transpose of the Matrix
    return np.matrix(base).getT()
def inverseMatrix(M):
	return np.linalg.inv(M)
def findpeak(a):
    result = []
    for i in range(len(a)):
        if a[i] > 10:
            result.append(i)
    return result
if __name__ == '__main__':

    signal_path = sys.argv[1]
    x = np.loadtxt(signal_path)
    N = len(x)
    # x = Ba so a = inverse(B) * x
    B = gen_basis(N)
    a = CosineTrans(x,B)
    frequency = []
    for index in (a[0]):
    	frequency.append(index)
    plot_ak(frequency, path = "./b06901160_freq.png")
    print(findpeak(frequency))
    mask = np.zeros((5,N))
    mask[0][78] = 1
    mask[2][352] = 1
    f1 = mask[0] * frequency
    f3 = mask[2] * frequency
    x1 = np.matmul(B,np.matrix(f1).getT())
    x3 = np.matmul(B,np.matrix(f3).getT())
    x1,x3 = np.array(x1),np.array(x3)
    x1,x3 = np.reshape(x1,(1,N)),np.reshape(x3,(1,N))
    x1,x3 = x1[0],x3[0]
    np.savetxt("b06901160_f1.txt",x1)
    np.savetxt("b06901160_f3.txt",x3)

