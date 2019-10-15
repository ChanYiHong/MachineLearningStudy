#Ex1-b
import numpy as np
data = np.array([46, 53, 29, 61, 36, 39, 47, 49, 52, 38, 55, 32, 57, 54, 44])
subset = np.array([data[0::2], data[1::2]])
for i in range(0,len(subset)):
    print("Mean, variance, and standard deviation for subset[%d] is:\n\t%.16f, %.16f, %.16f"
        %(i, subset[i].mean(), subset[i].var(), subset[i].std()))
    
#Ex2-b
import matplotlib.pyplot as plt
plotData = [subset[0], subset[1]]
plt.boxplot(plotData)
plt.show()
