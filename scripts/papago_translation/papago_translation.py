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

# ANSI escape sequence 제거 정규식 (로그 파일에 컬러 코드가 남지 않도록)
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def log_print(message):
    """터미널에 메시지를 출력하고, ANSI 컬러 코드를 제거한 메시지를 로그 파일에 기록"""
    print(message)
    clean_message = ANSI_ESCAPE.sub('', message)
    logging.info(clean_message)

class PapagoTranslator:
    def __init__(self, client_id, client_secret, source='ru', target='en', delay=0.3):
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
        """Papago Translation API를 이용해 텍스트를 번역"""
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
        """에러 발생 시 재시도하여 번역 진행"""
        for attempt in range(retries):
            result = self.translate_text(text)
            if result is not None:
                return result
            else:
                log_print(f"{Fore.YELLOW}Retry {attempt+1}/{retries} for text: {text[:30]} ...")
                time.sleep(retry_delay)
        return None

def process_json_file(file_path, translator):
    """단일 JSON 파일을 읽어 제품명, 설명, 스펙, 카테고리 번역 후 brand는 그대로 반환"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 제품명과 설명 처리 (최대 5000자 체크)
    product_name = data.get("name", "")
    description = data.get("description", "")
    if len(product_name) > 5000:
        log_print(f"{Fore.MAGENTA}Warning: 제품명 길이 초과 in {file_path}")
        product_name = product_name[:5000]
    if len(description) > 5000:
        log_print(f"{Fore.MAGENTA}Warning: 제품 설명 길이 초과 in {file_path}")
        description = description[:5000]
    
    translated_name = translator.translate_text_with_retry(product_name)
    time.sleep(translator.delay)
    translated_description = translator.translate_text_with_retry(description)
    time.sleep(translator.delay)
    
    # specifications 처리: 각 카테고리와 그 하위의 term, definition 번역
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

    # category 번역, brand는 그대로 유지
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
    """input_dir 내의 모든 JSON 파일에 대해 번역 후, translated/ 폴더에 저장하고 결과와 소요시간 출력"""
    output_dir = os.path.join(input_dir, "translated")
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)
            if os.path.isdir(file_path):
                continue
            log_print(f"{Fore.CYAN}Processing {filename} ...")
            start_time = time.time()  # 번역 시작 시간
            translated_result = process_json_file(file_path, translator)
            elapsed_time = time.time() - start_time  # 소요 시간 계산
            output_file_path = os.path.join(output_dir, filename)
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                json.dump(translated_result, out_f, indent=2, ensure_ascii=False)
            log_print(f"{Fore.GREEN}Saved translated file to {output_file_path}")
            log_print(f"{Fore.MAGENTA}Translation time for {filename}: {elapsed_time:.2f} seconds")
            # 번역된 결과 출력 (pretty-print)
            log_print(f"{Fore.BLUE}Translated result for {filename}:")
            colored_json = json.dumps(translated_result, indent=2, ensure_ascii=False)
            log_print(f"{Fore.LIGHTYELLOW_EX}{colored_json}{Style.RESET_ALL}\n")
            
def main():
    parser = argparse.ArgumentParser(
        description="Papago Translation: 번역할 JSON 파일들이 들어 있는 디렉토리(version0)를 처리합니다."
    )
    parser.add_argument("--client-id", required=True, help="Papago API Client ID")
    parser.add_argument("--client-secret", required=True, help="Papago API Client Secret")
    parser.add_argument("--input-dir", required=True, help="번역할 JSON 파일들이 위치한 디렉토리 (예: version0)")
    parser.add_argument("--source", default="ru", help="원본 언어 코드 (기본: ru)")
    parser.add_argument("--target", default="en", help="번역 대상 언어 코드 (기본: en)")
    parser.add_argument("--delay", type=float, default=0.1, help="API 호출 간 딜레이 (초 단위, 기본: 0.3)")
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