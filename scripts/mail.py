import requests
from bs4 import BeautifulSoup
import re

# List of domains to scrape
domains = ['https://www.linkedin.com/in/lady-portocarrero-mesia-395bb157/overlay/contact-info/']

# Regular expression for matching email addresses
email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# Function to fetch emails from a given URL
def fetch_emails(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all text that matches the email regular expression
        emails = set(email_regex.findall(soup.get_text()))
        return emails
    except Exception as e:
        print(f"An error occurred while fetching emails from {url}: {e}")
        return set()

# Main function to iterate over the domains and scrape emails
def main():
    all_emails = set()
    for domain in domains:
        print(f"Scraping emails from {domain}...")
        emails = fetch_emails(f'http://{domain}')
        all_emails.update(emails)
        for email in emails:
            print(email)
    
    print("\nAll found emails:")
    for email in all_emails:
        print(email)

if __name__ == '__main__':
    main()
