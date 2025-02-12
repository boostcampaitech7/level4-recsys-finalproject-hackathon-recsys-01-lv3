import requests
from pprint import pprint

url = "https://clovastudio.stream.ntruss.com/testapp/v1/tasks/fv49n0yw/chat-completions"

headers = {
    "Authorization": "Bearer <api-key>",
    "X-NCP-CLOVASTUDIO-REQUEST-ID": "",
    "Content-Type": "application/json; charset=utf-8",
    # 스트리밍 응답을 원한다면 아래 헤더를 추가:
    # "Accept": "text/event-stream"
}

data = {
    "messages": [
        {"role": "system", "content": "당신은 이커머스 로그 기반 프로모션 전략을 제안하는 AI 어시스턴트입니다."},
        {"role": "user", "content": "- 상품 ID: 441\n- 현재 가격: 755\n- 브랜드 ID: 726\n- 카테고리 ID: 33\n\n추천된 최적의 프로모션 값: 할인율 0.2, 추천 대상 유저 수 650, 최적 보상 3000\n사용자가 선택한 값: 할인율 0.25, 추천 대상 유저 수 350\n예상 보상: 2300\n\n이 데이터를 바탕으로 예상 보상 계산 및 전략 제안을 해줘."}
    ],
    "temperature": 0.01,
    "topK": 0,
    "topP": 0.8,
    "repeatPenalty": 5.0,
    "maxTokens": 1000,
    "includeAiFilters": True,
    "stopBefore": []
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
pprint(result)
