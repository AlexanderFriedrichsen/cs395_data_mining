age = [23,23,27,27,39,41,47,49,50,52,54,54,56,57,58,58,60,61]

fat = [9.5,26.5,7.8,17.8,31.4,25.9,27.4,27.2,31.2,34.6,42.5,28.8,33.4,30.2,34.1,32.9,41.2,35.7]
import numpy
print(numpy.std(age))
print(numpy.median(age))

import matplotlib.pyplot as plt

# plt.boxplot(age)
# plt.ylabel("age")
# plt.show()

# plt.boxplot(fat)
# plt.ylabel("fat%")
# plt.show()

plt.scatter(fat, age)
plt.ylabel("fat%")
plt.xlabel("age")
plt.show()

from scipy.stats import probplot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

probplot(age, plot=ax1)
ax1.set_title('Q-Q plot of ages')

probplot(fat, plot=ax2)
ax2.set_title('Q-Q plot of body fat percentages')

plt.show()