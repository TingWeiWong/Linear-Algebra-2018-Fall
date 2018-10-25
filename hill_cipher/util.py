import numpy as np

# key should be a numpy array
def inv_key(key):
    det = int(round(np.linalg.det(key)))
    inv = np.linalg.inv(key)
    tem = (inv * det)
    modinv = np.mod(det ** 29, 31)
    return np.around(np.mod(tem * modinv, 31)).astype(int)

# cipher1 = "_CUKZFBMKDFJ.KO"
cipher1_plain = [26,2,20,10,25,5,1,12,10,3,5,9,27,10,14]
cipher1 = np.reshape(cipher1_plain,(3,5))
key1 = [0,18,2,6,6,20,24,0,15]
key1 = np.reshape(key1,(3,3))
inverse_key1 = inv_key(key1)
original = np.matmul(inverse_key1,cipher1)
for i in range(3):
	for j in range(5):
		original[i][j] = original[i][j]%31
print ("Plain text1 = ",(original))
cipher2 = [21,29,3,13,20,4,24,0,7]
cipher2 = np.reshape(cipher2,(3,3))
plain2 = [14,7,26,28,26,18,7,8,19]
plain2 = np.reshape(plain2,(3,3))
inverse_plain2 = inv_key(plain2)
key2 = np.matmul(cipher2,inverse_plain2)
for i in range(3):
	for j in range(3):
		key2[i][j] = key2[i][j]%31
print ("Key2 = ",key2)
inv_key2 = inv_key(key2)
other_cipher = [6,0,7,19,30,28,30,15,0]
other_cipher = np.reshape(other_cipher,(3,3))
other_plain = np.matmul(inv_key2,other_cipher)
for i in range(3):
	for j in range(3):
		other_plain[i][j] = other_plain[i][j]%31
print (other_plain)


cipher_test1 = [21,9,22,20,21,28,4,3,8]
cipher_test1 = np.reshape(cipher_test1,(3,3))
key_test1 = [25,8,25,9,9,16,28,21,18]
key_test1 = np.reshape(key_test1,(3,3))

inverse_test_key = inv_key(key_test1)
plain_test1 = np.matmul(inverse_test_key,cipher_test1)
for i in range(3):
	for j in range(3):
		plain_test1[i][j] = plain_test1[i][j] %31
print ("Test Plain = ",(plain_test1))






