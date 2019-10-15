#Ex1-c
import numpy as np
D = np.random.randint(0,10,10000)
print("D =", D)
digit = np.arange(0,10)
count = np.zeros(10, dtype='int')
for i in digit:
    count[i] = (D == i).sum()
    print("{} appears {} times".format(i,count[i]))

#Ex2-c
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [8,6]
plt.pie(count, labels=digit, autopct='%.2f%%')
plt.show()
