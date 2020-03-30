import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

temp = pd.read_csv("data.csv",delimiter=",",names=["math","cs"])

first = np.asarray(temp["math"])
second = np.asarray(temp["cs"])


def gradientDescent():
    # initializing co-eff and intercept to 0
    m = 0
    c = 0
    # costvar to check for minimum cost and stop the iterations
    costvar = 0.000
    # learning rate variable to control the algorithm
    learningrate = 0.0002
    for i in range(1000000):
        # m - derivative for updating value of m
        md = (-2/len(first)) * sum((second - m * first - c) * first)
        # c - derivative for updating value of c
        cd = (-2/len(first)) * sum((second - m * first - c))
        # updating value of m and c using learning rate and derivative calculated
        m -= learningrate * md
        c -= learningrate * cd
        cost = (sum((second - (m * first + c))**2))/len(first)
        # comparison of cost for finding minumum possible cost
        if math.isclose(cost, costvar, rel_tol=1e-20):
            break
        # updating old costvar variable to new cost which is smaller than previous one
        costvar = cost

    print("Using gradient descent m = {} c= {} cost = {}".format(m,c,costvar))
    plt.plot(first, (m * first + c),color = 'red')
    return m,c


if __name__ == '__main__' :
    print("Checking for optimal values")
    gradientDescent()
    plt.xlabel('Maths')
    plt.ylabel('Computer science')
    plt.scatter(first, second, color='black')
    plt.savefig("gradientDescent.png")
