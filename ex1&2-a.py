#Ex1-a
import numpy as np
humidity = np.array([46, 53, 29, 61, 36, 39, 47, 49, 52, 38, 55, 32, 57, 54, 44])
moisture = np.array([12, 15,  7, 17, 10, 11, 11, 12, 14,  9, 16,  8, 18, 14, 12])
xBar = humidity.mean()
yBar = moisture.mean()
n = len(humidity)
B_numerator   = (humidity * moisture).sum() - n * xBar * yBar
B_denominator = (humidity * humidity).sum() - n * xBar * xBar
B = B_numerator / B_denominator
A = yBar - B * xBar
print("X = {}\nY = {}".format(humidity, moisture))
print("Linear regression (y = Bx + A):")
print("B = {}, A = {}".format(B, A))

#Ex2-a
import matplotlib.pyplot as plt
plt.scatter(humidity, moisture)
plt.xlabel('Humidity')
plt.ylabel('Moisture')
px = np.array([humidity.min()-1, humidity.max()+1])
py = B * px + A
plt.plot(px,py,color='r')
plt.show()
