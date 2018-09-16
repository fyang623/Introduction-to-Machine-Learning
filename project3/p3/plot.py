"""
File: plot.py

Code to to generate the plots for the project report.
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([2, 4, 8, 15, 30, 59, 118, 235, 469, 938, 1875, 3750, 7500, 15000, 30000, 60000])
y1 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,0.9764,0.937633,0.938])
y2 = np.array([0.1093, 0.146, 0.2723, 0.3438, 0.371, 0.5743,0.7, 0.7903, 0.8192, 0.8437, 0.8594, 0.847, 0.8491, 0.8595, 0.8827, 0.9168])

plt.plot(x, y1, label="Train")
plt.plot(x, y2, label="Test")


# Now add the legend with some customizations.
legend = plt.legend(loc='center right', shadow=True)

# # set the legend font size
# for label in legend.get_texts():
#     label.set_fontsize('small')

# # set the legend line width
# for label in legend.get_lines():
#     label.set_linewidth(1.5)

plt.xscale('log')
plt.xlabel("number of examples seen")
plt.ylabel("accuracy of the classifier")
plt.title('accuracy of the classifier vs. number of examples seen')
plt.grid(True)
plt.savefig('fig')
plt.show()

# x = np.array([6.85, 9.7, 11.65, 20.75, 21.55, 27.25, 15.50, 18.50, 7.25, 7.25, 7.25, 6.85, 7.05, 31.25, 12.0, 14.7, 11.95])
# y = np.array([6.85, 9.85, 13.2, 25.55, 27.20, 33.00, 18.55, 18.05, 8.55, 8.20, 8.45, 7.90, 7.75, 35.00, 15.0, 19.2, 16.6])
#
# m, b = np.polyfit(x, y, 1)
# plt.scatter(x, y, s=40)
# plt.plot(x, m*x + b, '-')
# plt.xlabel("number of moves using LAO*")
# plt.ylabel("number of moves using UCT")
# plt.title('UCT vs. LAO*')
# plt.grid(True)
# plt.savefig("fig5")
# plt.show()
