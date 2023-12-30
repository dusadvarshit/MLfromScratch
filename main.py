import streamlit as st

st.title("Machine Learning from Scratch")

st.subheader("Following are the algorithms that will be covered")

st.divider()

st.markdown("1. K Nearest Neighbor")
st.markdown("2. Linear Regression")
st.markdown("3. Logistic Regression")
st.markdown("4. Decision Trees")
st.markdown("5. Random Forest")
st.markdown("6. Naive Bayes")
st.markdown("7. Principal Component Analysis")
st.markdown("8. Perceptron")
st.markdown("9. Support Vector Machine")
st.markdown("10. K-Means Clustering")


number = st.number_input('Insert a number')
st.write('The current number is ', number)