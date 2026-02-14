import streamlit as st

# Title
st.title('DealFinder Application')

# Sidebar
st.sidebar.header('User Input')

# User input fields
budget = st.sidebar.number_input('Budget', min_value=0, value=500)
location = st.sidebar.text_input('Location', 'New York')
deal_type = st.sidebar.selectbox('Deal Type', ['Discount', 'Free Shipping', 'Buy One Get One'])

# Main content
st.header('Deals')
st.write(f'Finding deals for budget: ${budget}, location: {location}, type: {deal_type}')

# Logic to find deals can be implemented here

# Display found deals (dummy placeholder)
deals = [{'deal': 'Example Deal 1', 'price': 100}, {'deal': 'Example Deal 2', 'price': 200}]
for deal in deals:
    st.write(f"{deal['deal']} - ${deal['price']}")
