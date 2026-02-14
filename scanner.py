import requests
import time

class BackgroundScanner:
    def __init__(self, telegram_bot_token, telegram_chat_id, ebay_api_key):
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.ebay_api_key = ebay_api_key

    def send_telegram_alert(self, message):
        url = f'https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage'
        payload = {'chat_id': self.telegram_chat_id, 'text': message}
        requests.post(url, data=payload)

    def ebay_item_search(self, keyword):
        url = 'https://api.ebay.com/buy/browse/v1/item_summary/search'
        headers = {'Authorization': f'Bearer {self.ebay_api_key}'}
        params = {'q': keyword}
        response = requests.get(url, headers=headers, params=params)
        return response.json()

    def run(self, keyword):
        while True:
            items = self.ebay_item_search(keyword)
            # Assuming we are looking for a specific item condition
            for item in items.get('itemSummaries', []):
                title = item.get('title')
                price = item.get('price', {}).get('value')
                alert_message = f'Found item: {title} at price: {price}'
                self.send_telegram_alert(alert_message)
            # Wait for a specified time before scanning again
            time.sleep(600)  # Runs every 10 minutes

# Example usage:
# scanner = BackgroundScanner('your_telegram_bot_token', 'your_chat_id', 'your_ebay_api_key')
# scanner.run('laptop')
