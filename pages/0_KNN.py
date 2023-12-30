import streamlit as st

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

st.divider()