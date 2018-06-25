import matplotlib.pyplot as plt

def plotData(X, y):
    plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
    plt.xlim(0,25)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()