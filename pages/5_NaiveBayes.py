import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

st.title("Naive Bayes")

st.subheader('Theory')
on = st.toggle(">")

## Writing Theory 
if on:
    st.write("""Naive Bayes is a Machine Learning based classification method
                that uses Bayes theorem and associated assumptions to predict
                class of input vector.
             """)
    
    st.write("""Makes naive assumption about independence between individual features.""")
    
    st.write("""Mathematically, the BAYES THEOREM is represented by:""")
    
    st.latex(r'''
    P(A|B) = \frac{P(B|A)*P(A)}{P(B)}
    ''')

    st.latex(r'''
    P(y|X) = \frac{P(X|y)*P(y)}{P(X)}
    ''')

    st.latex(r'''
    P(y|X) = \frac{P(x_{1}|y)*P(x_{2}|y)*P(x_{3}|y)....P(x_{n}|y)*P(y)}{P(X)}
    ''')

    st.latex(r'''
    argmax(P(y_{1}/X), P(y_{2}/X),.... P(y_{n}/X))
    ''')

    st.write("""Since P(X) is a common denominator in the calculations, 
    we remove P(X) from calculations as it doesn't affect the argmax function.""")

    st.write("""Also because product of probabilities can be a very small number 
    -that can hurt the floating point operations- we will indeed use log(P) which 
    will provide the same qualitative response.""")

    st.write("""Also for reducing the code complexity we will be 
                using numpy array broadcasting.""")

st.divider()

def accuracy(y_true, y_pred):
  accuracy = np.sum(y_true == y_pred) / len(y_true)
  return accuracy

class NaiveBayes:

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self._classes = np.unique(y)
    n_classes = len(self._classes)

    # Calculate mean, var, and prior for each class
    self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
    self._var = np.zeros((n_classes, n_features), dtype = np.float64)
    self._priors = np.zeros(n_classes, dtype=np.float64)

    for idx, c in enumerate(self._classes):
      X_c = X[y==c]
      self._mean[idx, :] = X_c.mean(axis=0)
      self._var[idx, :] = X_c.var(axis=0)
      self._priors[idx] = X_c.shape[0] / float(n_samples) # P(y) Frequency of each class in data

  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)

  def _predict(self, x):
    posteriors = []

    # Calculate posterior probability for each class, 
    ## P(X) -> Ignored from calculation because it is a common denominator and doesn't
    ## affect argmax
    for idx, c in enumerate(self._classes):
      prior = np.log(self._priors[idx]) # log(P(y))
      posterior = np.sum(np.log(self._pdf(idx, x))) ## log(P(x1|y)) + log(P(x2|y)) ....
      posterior = posterior + prior ## posterior + log(P(y)) {prior}
      posteriors.append(posterior)

    # Return class with the highest posterior
    return self._classes[np.argmax(posteriors)]

  def _pdf(self, class_idx, x):
    mean = self._mean[class_idx]
    var = self._var[class_idx]
    numerator = np.exp(-((x-mean)**2) / (2*var))
    denominator = np.sqrt(2 *np.pi*var)

    return numerator / denominator

X, y = datasets.make_classification(
    n_samples=1000, n_features=10, n_classes=2, random_state=123
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

st.write("""We are working with sk-learn classification dataset. 
        It has 1000 samples, 10 features
        and 2 labels.
             """)

st.write("Naives Bayes classification accuracy", accuracy(y_test, predictions))

