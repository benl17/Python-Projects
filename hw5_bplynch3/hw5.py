import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == "__main__":
    csvData = pd.read_csv(sys.argv[1])
    X = csvData.year
    Y = csvData.days
    plt.plot(X, Y)
    plt.ylabel("Number of frozen days")
    plt.xlabel("Year")
    plt.xticks(np.arange(min(X), max(X)+1, 1.0))
    #Q2
    plt.savefig("plot.jpg")

    #Q3a
    print("Q3a:")
    xI = np.vstack([np.ones(len(X)), X])
    xI = xI.astype(int)
    X = np.transpose(xI)
    print(X)

    #Q3b
    Y = Y.to_numpy(dtype=np.int64)
    print("Q3b:")
    print(Y)

    #Q3c
    print("Q3c:")
    Z = np.dot(xI, X)
    print(Z)

    #Q3d
    print("Q3d:")
    I = np.linalg.inv(Z)
    print(I)

    #Q3e
    print("Q3e:")
    PI = np.dot(I, xI)
    print(PI)

    #Q3f
    print("Q3f:")
    hat_beta = np.dot(PI, Y)
    print(hat_beta)

    #Q4
    y_test = (2022 * hat_beta[1]) + hat_beta[0]
    print("Q4: " + str(y_test))

    #Q5a
    if hat_beta[1] > 0:
        print("Q5a: >")
        print("Q5b: This means that our slope is increasing thus the number of days Lake Mendota is frozen is also increasing every year.")
    elif hat_beta[1] < 0:
        print("Q5a: <")
        print("Q5b: This means that our slope is decresing thus the number of days Lake Mendota is frozen is also decreasing every year.")
    else:
        print("Q5a: =")
        print("Q5b: This means that our slope is constant thus the number of days Lake Mendota will stay the same as previous year.")

    #Q6a
    xStar = 0-(hat_beta[0]/hat_beta[1])
    print("Q6a:",xStar)

    #Q6b
    print("Q6b: I believe that this is a compelling prediction because based on plot.jpg and the data given to us, it clearly shows that", 
    "the number of days that Lake Mendota","\n","is frozen has been decreasing for over 150 years.")