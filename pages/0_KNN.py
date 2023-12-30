import streamlit as st

st.button("Reset")
if st.button('Say hello ✅'):
    st.write('Why hello there')
else:
    st.write('Goodbye')


a = st.button("Yayay")
st.write(a)

st.write("$ a' \\beta $")


animal_shelter = ['cat', 'dog', 'rabbit', 'bird']

animal = st.text_input('Type an animal')

if st.button('Check availability'):
    have_it = animal.lower() in animal_shelter
    'We have that animal!' if have_it else 'We don\'t have that animal.'