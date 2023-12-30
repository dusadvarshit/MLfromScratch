import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

st.title("Linear Regression")

st.subheader('Theory')
on = st.toggle(">")

## Writing Theory 
if on:
    st.write("""Linear Regression is a Machine Learning based modeling technique where 
             a independent variable is modelled as a weighted sum of one or more dependent variable.
             """)
    
    st.write("""The model is considered to be linear. This means for a specific change 
             in one unit of $$X$$, we can determine it's impact on $$y$$.""")
    
    st.write("""Mathematically, the linear regression is represented by:""")
    
    st.latex(r'''
    f(x) = \beta_{0} + \beta_{0}X_{1} + \beta_{0}X_{2} .... + \beta_{0}X_{p}
    ''')

    st.write("""Do note that Linear Regression is almost always an approximation 
             of the real world.""")
    
    st.write("""The quality of the fit of a Linear Regression model is measured by:""")

    st.latex(r'''
    MSE = \frac{1}{n}\Sigma_{i=1}^{n}(y_{i} - \hat{f}(x_{i}))^2
    ''')

    st.write("""Notice the little hat symbol on f: $$\hat{f}$$ - it represents
             that we only have an estimate of $$f$$. Not it's exact function. 
             Getting exact function is almost impossible in real world.""")

st.divider()


### Logic Code

class LinearRegression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
def mse(y_test, predictions):
  mse = np.mean((y_test-predictions)**2)
  return mse

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


reg = LinearRegression()
reg.fit(X_train, y_train)
prediction_train = reg.predict(X_train)
predictions = reg.predict(X_test)

mse = mse(y_test, predictions)


fig, ax = plt.subplots()
ax.scatter(X_train[:, 0], y_train, color = "b", marker = "o", s = 30)
ax.plot(X_train[:, 0], prediction_train, color = "r",)
ax.set_xlabel("Independent Variable")
ax.set_ylabel("Dependent Variable")
ax.set_title("Training Raw Data and it's Linear Regression")

st.pyplot(fig)


fig, ax = plt.subplots()
ax.scatter(X_test[:, 0], y_test, color = "b", marker = "o", s = 30)
ax.plot(X_test[:, 0], predictions, color = "r",)
ax.set_xlabel("Independent Variable")
ax.set_ylabel("Dependent Variable")
ax.set_title("Test Raw Data and it's Linear Regression")

st.pyplot(fig)

st.subheader(f"Mean Squared Error: {np.round(mse, 2)}")
