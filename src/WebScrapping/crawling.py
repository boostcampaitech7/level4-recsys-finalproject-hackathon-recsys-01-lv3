import os
import sys
import random
import polars as pl
import aiohttp
import asyncio
from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup

BASE_URL = "https://kaspi.kz/shop/p/"
PRODUCT_ID_FILE = "../../Hackton/product_id_part_9"
HTML_OUTPUT_DIR = "../../Hackton/html/"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.67 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
]

if not os.path.exists(HTML_OUTPUT_DIR):
    os.makedirs(HTML_OUTPUT_DIR)

def load_completed_ids():
    """
    Load the IDs of products whose HTML files have already been saved.

    Returns:
        set: A set of completed product IDs.
    """
    if os.path.exists(HTML_OUTPUT_DIR):
        completed_files = os.listdir(HTML_OUTPUT_DIR)
        completed_ids = {file.split(".")[0] for file in completed_files if file.endswith(".html")}
        return completed_ids
    return set()

async def save_html_to_file(product_id, html_content):
    """
    Save the cleaned HTML content to a file.

    Args:
        product_id (str): The ID of the product.
        html_content (str): The cleaned HTML content to save.
    """
    file_path = os.path.join(HTML_OUTPUT_DIR, f"{product_id}.html")
    try:
        with open(file_path, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
        print(f"HTML for product {product_id} saved successfully.")
    except Exception as e:
        print(f"Error saving HTML for product {product_id}: {e}")

def clean_html(html_content):
    """
    Clean the HTML content by removing unnecessary script tags.

    Args:
        html_content (str): The raw HTML content.

    Returns:
        str: The cleaned HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    for script in soup.find_all('script'):
        script_str = str(script)
        if ("BACKEND.config" in script_str or 
            "BACKEND.components.citySelection" in script_str or 
            "window.sentryConfig" in script_str or 
            "getCookie" in script_str):
            script.decompose()

    return str(soup)

async def fetch_and_save_html(session: ClientSession, product_id: str, retries: int = 3):
    """
    Fetch the HTML content of a product page and save it after cleaning.

    Args:
        session (ClientSession): The aiohttp session object.
        product_id (str): The ID of the product to fetch.
        retries (int): Number of retry attempts in case of failure.
    """
    url = f"{BASE_URL}-{product_id}/"
    attempt = 0
    
    while attempt < retries:
        try:
            user_agent = random.choice(USER_AGENTS)
            headers = {"User-Agent": user_agent}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html_content = await response.text()
                    cleaned_html = clean_html(html_content)
                    await save_html_to_file(product_id, cleaned_html)
                    return
                else:
                    print(f"Failed to fetch {product_id}: HTTP {response.status}")
                    return
        except asyncio.TimeoutError:
            print(f"Timeout occurred while fetching {product_id}, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            print(f"Error fetching {product_id}: {e}")
            return
        
        wait_time = random.uniform(2.0, 3.0)
        await asyncio.sleep(wait_time)
        attempt += 1

    print(f"Failed to fetch {product_id} after {retries} attempts.")

async def main():
    """
    Main function to orchestrate fetching and saving HTML for all products.
    
    Reads product IDs from a Parquet file and processes them asynchronously.
    """
    product_ids = pl.read_parquet(PRODUCT_ID_FILE)['product_id'].to_list()
    completed_ids = load_completed_ids()
    remaining_ids = [pid for pid in product_ids if str(pid) not in completed_ids]

    timeout = ClientTimeout(total=10, connect=5)
    async with ClientSession(timeout=timeout) as session:
        tasks = []
        for idx, pid in enumerate(remaining_ids, start=1):
            print(f"Processing product {idx}/{len(remaining_ids)}: {pid}")
            tasks.append(fetch_and_save_html(session, pid))
            wait_time = random.uniform(2.0, 2.2)
            if len(tasks) >= 1:
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(wait_time)

        if tasks:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
