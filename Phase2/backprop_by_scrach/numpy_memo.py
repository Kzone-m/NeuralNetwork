import numpy as np

lst1 = [1, 2, 3]
lst2 = [
    [1, 2, 3],
    [3, 4, 5]
]
t1 = np.array([lst1])
t2 = np.array([lst2])

print(t1.shape[0]) # 1
print(t2.shape[0]) # 1
print(t2.shape[1]) # 2
print(t2.shape[2]) # 3


print(np.sum(t1, axis=0))
print(np.sum(t2, axis=0))

#print(t1.shape)
print(t1.transpose())


print(t1 * t2)

print(np.dot(np.array([[1,2]]).T, np.array([[3,4]])))


lst3 = [1,2]
lst4 = [
    [4, 5],
    [4, 5]
]

s1 = np.array(lst3)
s2 = np.array(lst4)

print(s2 - s1)


var = np.array([[  1.41657887e+03,  2.48313468e+35]])
print(2.48313468e+35 - 1.41657887e+33)

print(np.dot(np.array([[1,2]]).T, np.array([[3,4]])))

print(s1[0])