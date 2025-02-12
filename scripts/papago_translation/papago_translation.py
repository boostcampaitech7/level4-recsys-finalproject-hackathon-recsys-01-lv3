#!/usr/bin/env python3
import os
import json
import time
import requests
import urllib.parse
import argparse
import logging
import re
from colorama import init, Fore, Style

# colorama 초기화 (autoreset=True를 지정하여 스타일이 자동 초기화됨)
init(autoreset=True)

# 로그 설정: papago_translation.log 파일에 기록 (append 모드)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="papago_translation.log",
    filemode="a"
)

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def log_print(message):
    """
    Print a message to the terminal and log it to a file after removing ANSI color codes.

    Args:
        message (str): The message to be printed and logged.
    """
    print(message)
    clean_message = ANSI_ESCAPE.sub('', message)
    logging.info(clean_message)

class PapagoTranslator:
    """
    A class to handle text translation using the Papago Translation API.
    """

    def __init__(self, client_id, client_secret, source='ru', target='en', delay=0.3):
        """
        Initialize the PapagoTranslator with API credentials and settings.

        Args:
            client_id (str): Papago API client ID.
            client_secret (str): Papago API client secret.
            source (str): Source language code. Default is 'ru'.
            target (str): Target language code. Default is 'en'.
            delay (float): Delay between API calls in seconds. Default is 0.3.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.source = source
        self.target = target
        self.delay = delay
        self.api_url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        self.headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/x-www-form-urlencoded"
        }

    def translate_text(self, text):
        """
        Translate a given text using the Papago Translation API.

        Args:
            text (str): The text to be translated.

        Returns:
            str: Translated text if successful, or None if an error occurs.
        """
        if not text:
            return ""
        data = {
            "source": self.source,
            "target": self.target,
            "text": text
        }
        encoded_data = urllib.parse.urlencode(data)
        try:
            response = requests.post(self.api_url, headers=self.headers, data=encoded_data)
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("result", {}).get("translatedText")
            else:
                log_print(f"{Fore.RED}Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            log_print(f"{Fore.RED}Exception occurred: {e}")
            return None

    def translate_text_with_retry(self, text, retries=3, retry_delay=1):
        """
        Translate a given text with retry logic in case of errors.

        Args:
            text (str): The text to be translated.
            retries (int): Number of retry attempts. Default is 3.
            retry_delay (float): Delay between retries in seconds. Default is 1.

        Returns:
            str: Translated text if successful, or None after all retries fail.
        """
        for attempt in range(retries):
            result = self.translate_text(text)
            if result is not None:
                return result
            else:
                log_print(f"{Fore.YELLOW}Retry {attempt+1}/{retries} for text: {text[:30]} ...")
                time.sleep(retry_delay)
        return None

def process_json_file(file_path, translator):
    """
    Process a single JSON file by translating product details and specifications.

    Args:
        file_path (str): Path to the JSON file to be processed.
        translator (PapagoTranslator): Translator instance for handling translations.

    Returns:
        dict: Translated product details and specifications.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    product_name = data.get("name", "")
    description = data.get("description", "")
    if len(product_name) > 5000:
        log_print(f"{Fore.MAGENTA}Warning: Product name length exceeds limit in {file_path}")
        product_name = product_name[:5000]
    if len(description) > 5000:
        log_print(f"{Fore.MAGENTA}Warning: Product description length exceeds limit in {file_path}")
        description = description[:5000]

    translated_name = translator.translate_text_with_retry(product_name)
    time.sleep(translator.delay)
    translated_description = translator.translate_text_with_retry(description)
    time.sleep(translator.delay)

    translated_specifications = {}
    specifications = data.get("specifications", {})
    for spec_category, entries in specifications.items():
        translated_spec_category = translator.translate_text_with_retry(spec_category)
        time.sleep(translator.delay)
        translated_entries = []
        for entry in entries:
            term = entry.get("term", "")
            definition = entry.get("definition", "")
            translated_term = translator.translate_text_with_retry(term)
            time.sleep(translator.delay)
            translated_definition = translator.translate_text_with_retry(definition)
            time.sleep(translator.delay)
            translated_entries.append({
                "term": translated_term,
                "definition": translated_definition
            })
        translated_specifications[translated_spec_category] = translated_entries

    category_value = data.get("category", "")
    translated_category_value = translator.translate_text_with_retry(category_value) if category_value else ""
    time.sleep(translator.delay)
    brand = data.get("brand", "")

    result = {
        "product_name": translated_name,
        "category": translated_category_value,
        "brand": brand,
        "description": translated_description,
        "specifications": translated_specifications
    }
    return result

def process_all_json_files(input_dir, translator):
    """
    Process all JSON files in the input directory and save translations to the output directory.

    Args:
        input_dir (str): Directory containing JSON files to be processed.
        translator (PapagoTranslator): Translator instance for handling translations.
    """
    output_dir = os.path.join(input_dir, "translated")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            if os.path.isdir(file_path):
                continue
            log_print(f"{Fore.CYAN}Processing {filename} ...")
            start_time = time.time()
            translated_result = process_json_file(file_path, translator)
            elapsed_time = time.time() - start_time
            output_file_path = os.path.join(output_dir, filename)
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                json.dump(translated_result, out_f, indent=2, ensure_ascii=False)
            log_print(f"{Fore.GREEN}Saved translated file to {output_file_path}")
            log_print(f"{Fore.MAGENTA}Translation time for {filename}: {elapsed_time:.2f} seconds")
            log_print(f"{Fore.BLUE}Translated result for {filename}:")
            colored_json = json.dumps(translated_result, indent=2, ensure_ascii=False)
            log_print(f"{Fore.LIGHTYELLOW_EX}{colored_json}{Style.RESET_ALL}\n")

def main():
    """
    Main function to handle command-line arguments and execute the translation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Papago Translation: Process JSON files in the input directory."
    )
    parser.add_argument("--client-id", required=True, help="Papago API Client ID")
    parser.add_argument("--client-secret", required=True, help="Papago API Client Secret")
    parser.add_argument("--input-dir", required=True, help="Directory containing JSON files to translate")
    parser.add_argument("--source", default="ru", help="Source language code. Default is 'ru'.")
    parser.add_argument("--target", default="en", help="Target language code. Default is 'en'.")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls in seconds. Default is 0.3.")
    
    args = parser.parse_args()

    translator = PapagoTranslator(
        client_id=args.client_id,
        client_secret=args.client_secret,
        source=args.source,
        target=args.target,
        delay=args.delay
    )

    process_all_json_files(args.input_dir, translator)

if __name__ == "__main__":
    main()
