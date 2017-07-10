import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

x = np.arange(16) - 2
y = x.reshape((4, 4))


print "y matrix: \n{}".format(y)
print "\nfirst column of each row "
print y[:, 0]
print "second column of each row "
print y[:, 1]

plt.scatter(y[:, 0], y[:, 1], color="g", marker='o', linewidths=40)

plt.show()


