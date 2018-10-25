import numpy as np

# key should be a numpy array
def inv_key(key):
    det = int(round(np.linalg.det(key)))
    inv = np.linalg.inv(key)
    tem = (inv * det)
    modinv = np.mod(det ** 29, 31)
    return np.around(np.mod(tem * modinv, 31)).astype(int)

def char_to_num(string):
	result = []
	for character in string:
		if character == "_":
			result.append(26)
		elif character == ".":
			result.append(27)
		elif character ==",":
			result.append(28)
		elif character == "?":
			result.append(29)
		elif character =="!":
			result.append(30)
		else:
			result.append(ord(character)-65)
	return result
def num_to_char(array):
	result = []
cipher1 = np.reshape(char_to_num("_CUKZFBMKDFJ.KO"),(3,5))
key1 = [0, 18, 2, 6, 6, 20, 24, 0, 15]
inverse_key1 = inv_key(np.reshape(key1,(3,3)))
plain1 = np.matmul(inverse_key1,cipher1)
print (plain1)
for i in range(3):
	for j in range(5):
		plain1[i][j] = plain1[i][j]%31
print (plain1)
'''
Assignment 
1. Cipher Text for problem 1 
_CUKZFBMKDFJ.KO

2. Key for problem 1
0 18 2 6 6 20 24 0 15

3.Cipher Text for problem 2 
V?DNUEYAH

4. Plain Text for problem 2 
OH_,_SHIT

5.The other Cipher Text
GAHT!,!PA

'''

