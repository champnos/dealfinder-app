# DealFinder

DealFinder is a Windows desktop application that uses the official eBay Browse API to perform structured product searches.

## Purpose

DealFinder helps identify live listings that match user-defined search profiles, including:

- Product model variations
- Buying option filters (Fixed Price / Auction)
- Condition filters
- Keyword inclusion / exclusion logic

The application is designed for structured product discovery and listing analysis.

## Architecture

- Windows desktop application (Streamlit-based UI)
- Background scanner process for scheduled searches
- Uses official eBay Browse API endpoints only
- Implements rate limiting and retry logic
- Tracks daily API usage

## API Usage

- Approx. 50–60 Browse API calls per scan
- Scheduled scans at controlled intervals
- Exponential backoff on 429 responses
- Daily API call monitoring

DealFinder complies with eBay's API License Agreement and does not perform scraping.

## Contact

For application growth or API review inquiries:

Email: champnos@hotmail.com
