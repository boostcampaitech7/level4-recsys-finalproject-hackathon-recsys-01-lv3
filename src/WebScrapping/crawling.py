import time
import os
import random
import polars as pl
import aiohttp
import asyncio
from aiohttp import ClientSession, ClientTimeout
from bs4 import BeautifulSoup  # BeautifulSoup 추가

# 설정
BASE_URL = "https://kaspi.kz/shop/p/"
PRODUCT_ID_FILE = "../../Hackton/product_id_part_9"
HTML_OUTPUT_DIR = "../../Hackton/html/"

# User-Agent 리스트 정의
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.67 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
]

# HTML 저장 디렉토리 생성
if not os.path.exists(HTML_OUTPUT_DIR):
    os.makedirs(HTML_OUTPUT_DIR)

# 완료된 product_id를 로드하는 함수
def load_completed_ids():
    if os.path.exists(HTML_OUTPUT_DIR):
        completed_files = os.listdir(HTML_OUTPUT_DIR)
        completed_ids = {file.split(".")[0] for file in completed_files if file.endswith(".html")}
        return completed_ids
    return set()

# HTML 파일 저장 함수
async def save_html_to_file(product_id, html_content):
    file_path = os.path.join(HTML_OUTPUT_DIR, f"{product_id}.html")
    try:
        with open(file_path, "w", encoding="utf-8") as html_file:
            html_file.write(html_content)
        print(f"HTML for product {product_id} saved successfully.")
    except Exception as e:
        print(f"Error saving HTML for product {product_id}: {e}")

# HTML 파싱 및 특정 스크립트 태그만 제거
def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # "BACKEND.config" 또는 "BACKEND.components.citySelection"을 포함하는 <script> 태그 삭제
    for script in soup.find_all('script'):
        script_str = str(script)
        if ("BACKEND.config" in script_str or 
            "BACKEND.components.citySelection" in script_str or 
            "window.sentryConfig" in script_str or 
            "getCookie" in script_str):
            script.decompose()

    # 수정된 HTML 반환
    return str(soup)

# 페이지 요청 및 HTML 저장 함수
async def fetch_and_save_html(session: ClientSession, product_id: str, retries: int = 3):
    url = f"{BASE_URL}-{product_id}/"
    attempt = 0
    
    while attempt < retries:
        try:
            # 랜덤한 User-Agent 선택
            user_agent = random.choice(USER_AGENTS)
            headers = {"User-Agent": user_agent}
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html_content = await response.text()
                    cleaned_html = clean_html(html_content)
                    await save_html_to_file(product_id, cleaned_html)
                    return  # 요청이 성공하면 함수 종료
                else:
                    print(f"Failed to fetch {product_id}: HTTP {response.status}")
                    return  # HTTP 오류 발생 시 종료
        except asyncio.TimeoutError:
            print(f"Timeout occurred while fetching {product_id}, retrying... ({attempt + 1}/{retries})")
        except Exception as e:
            print(f"Error fetching {product_id}: {e}")
            return  # 다른 오류 발생 시 종료
        
        # 재시도 전 대기 시간
        wait_time = random.uniform(2.0, 3.0)  # 예: 2초에서 3초 사이의 대기 시간
        await asyncio.sleep(wait_time)
        attempt += 1

    print(f"Failed to fetch {product_id} after {retries} attempts.")

# 비동기 작업을 관리하는 함수
async def main():
    # 데이터 불러오기
    product_ids = pl.read_parquet(PRODUCT_ID_FILE)['product_id'].to_list()
    completed_ids = load_completed_ids()
    remaining_ids = [pid for pid in product_ids if str(pid) not in completed_ids]

    # aiohttp 클라이언트 세션 생성
    timeout = ClientTimeout(total=10, connect=5)
    async with ClientSession(timeout=timeout) as session:
        tasks = []
        for idx, pid in enumerate(remaining_ids, start=1):
            print(f"Processing product {idx}/{len(remaining_ids)}: {pid}")
            tasks.append(fetch_and_save_html(session, pid))
            wait_time = random.uniform(2.0, 2.2)  # 요청 간 대기 시간 (2초 ~ 2.2초)
            if len(tasks) >= 1:  # 한 번에 최대 1개의 요청을 보내도록 조정
                await asyncio.gather(*tasks)
                tasks = []
                await asyncio.sleep(wait_time)  # 각 배치 후 대기 시간

        # 남은 요청 처리
        if tasks:
            await asyncio.gather(*tasks)

# 프로그램 실행
if __name__ == "__main__":
    asyncio.run(main())
