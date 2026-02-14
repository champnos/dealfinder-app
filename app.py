import streamlit as st

# Title of the app
st.title("My Streamlit App")

# Input field for user
user_input = st.text_input("Enter some text")

# Button to submit
if st.button('Submit'):
    st.write(f'You entered: {user_input}')
    
# Displaying a simple line chart
st.line_chart([1, 2, 3, 4, 5])