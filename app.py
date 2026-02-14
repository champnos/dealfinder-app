import streamlit as st
import requests

# Sample product database
products = [
    {"name": "Rare Item 1", "price": 100, "available": True},
    {"name": "Common Item", "price": 10, "available": True},
    {"name": "Rare Item 2", "price": 200, "available": False},
]

# User profiles (for demonstration purposes)
user_profiles = {
    "user1": {"email": "user1@example.com", "alerts": []},
    "user2": {"email": "user2@example.com", "alerts": []},
}

# Function to scan for rare items
def scan_for_rare_items():
    return [prod for prod in products if prod["price"] > 150 and prod["available"]]

# Function to send Telegram alerts
def send_telegram_alert(item):
    api_url = "https://api.telegram.org/bot<YOUR_TELEGRAM_BOT_TOKEN>/sendMessage"
    chat_id = "<YOUR_CHAT_ID>"
    message = f"Alert: New rare item found - {item['name']} at {item['price']}"
    requests.post(api_url, data={"chat_id": chat_id, "text": message})

# Streamlit UI
st.title("DealFinder Application")
st.subheader("Scan for Rare Items")

if st.button("Scan for Rare Items"):
    rare_items = scan_for_rare_items()
    if rare_items:
        for item in rare_items:
            st.write(f"Found rare item: {item['name']} at price ${item['price']}")
            send_telegram_alert(item)
    else:
        st.write("No rare items found.")

st.sidebar.header("User Profiles")
selected_user = st.sidebar.selectbox("Select User", list(user_profiles.keys()))

if selected_user:
    user_data = user_profiles[selected_user]
    st.sidebar.write(f"Email: {user_data['email']}")
    st.sidebar.write("Alerts:")
    for alert in user_data["alerts"]:
        st.sidebar.write(alert)
