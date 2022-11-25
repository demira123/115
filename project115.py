import pandas as pd
import plotly.express as pe
import matplotlib.pyplot as mlp
from sklearn.linear_model import LogisticRegression
import numpy as np


data = pd.read_csv("C115/project115.csv")

velocity=data["Velocity"].tolist()

escaped=data["Escaped"].tolist()


# --------------------------- using the classifier Logistic Regression ------------------------------------


X = np.reshape(velocity , (len(velocity) , 1) )
Y = np.reshape(escaped, (len(escaped), 1) )

lr = LogisticRegression()

lr.fit(X,Y)

mlp.figure()
mlp.scatter(X.ravel() , Y , color="black" , zorder = 20)

def model(x):
    return 1 / (1 + np.exp(-x))


X_test = np.linspace(0,100,200)

chances = model( X_test * lr.coef_ + lr.intercept_).ravel()



mlp.plot(X_test, chances, color='red', linewidth=3)

mlp.axhline(y=0, color='k', linestyle='-')
mlp.axhline(y=1, color='k', linestyle='-')
mlp.axhline(y=0.5, color='b', linestyle='--')

mlp.axvline(x=X_test[165], color='b', linestyle='--')

mlp.ylabel('y')
mlp.xlabel('X')
mlp.xlim(75, 85)
mlp.show()


# -------------------- will ask the user to enter the velocity based on which will predict --------------------- 

userScore = float(input("Enter your velocity here!"))

chances = model( userScore * lr.coef_ + lr.intercept_).ravel()


if chances >= 1 :
    print("The space object will escape!")
elif chances <= 0.01  :
    print("The space object will not escape!")
elif chances <= 0.5 :
    print("The space object might not escape!")
else:
    print("The space object may escape!")





























