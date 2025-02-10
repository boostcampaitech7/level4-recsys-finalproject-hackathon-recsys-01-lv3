import os
import sys
project_root = os.path.join(os.path.expanduser("~"), "Hackathon")
if project_root not in sys.path:
    sys.path.append(project_root)
    
import json
import requests
from pprint import pprint

json_dir = os.path.join(project_root, "src/data/llm_input/rl_json")
with open(os.path.join(json_dir, "111029.json"), "r", encoding="utf-8") as f:
    product_data = json.load(f)
    
system_prompt = (
    "당신은 이커머스 로그 및 강화학습 모델의 출력을 바탕으로,\n"
    "각 상품별 최적 할인율(action1)과 추천 대상 유저 수(action2),\n"
    "그리고 예상 보상을 계산해주는 AI 어시스턴트입니다.\n\n"
    "사용자가 할인율이나 대상 유저 수를 달리 지정했을 때,\n"
    "1) 예상 판매 가격,\n"
    "2) 구매 전환율,\n"
    "3) 최종 보상 계산 근거,\n"
    "4) 전략 제안\n"
    "등을 단계적으로 설명해주어야 합니다.\n\n"
    "답변 시:\n"
    "- 가격과 보상은 화폐 단위 '원'을 의미하지 않습니다. 화폐 단위로 표현하지 마세요."
    "- 할인율과 가격, 대상 유저 수 등 숫자 데이터를 명확히 표시하세요.\n"
    "- 계산 과정이나 추론 과정을 간략히 서술하여 이해를 돕습니다.\n"
    "- 전략 제안 시, 할인율 변경이 브랜드 가치와 이익률에 미치는 영향도 함께 고려하세요.\n"
    "- 지나치게 짧은 답변보다는 상세하고 구체적인 문장을 사용해주세요."
)

user_content = (
    f"- 상품 ID: {product_data['product_id']}\n"
    f"- 현재 가격: {product_data['current_price']}\n"
    f"- 브랜드 ID: {product_data['brand_id']}\n"
    f"- 카테고리 ID: {product_data['category_id']}\n\n"
    f"추천된 최적의 프로모션 값: 할인율 {product_data['recommended_promotion']['discount_rate']}, "
    f"추천 대상 유저 수 {product_data['recommended_promotion']['target_users']}, "
    f"최적 보상 {product_data['recommended_promotion']['optimal_reward']}\n"
    f"사용자가 선택한 값: 할인율 {product_data['user_selected']['discount_rate']}, "
    f"추천 대상 유저 수 {product_data['user_selected']['target_users']}\n"
    f"예상 보상: {product_data['expected_reward']}\n\n"
    "이 데이터를 바탕으로 예상 보상 계산 및 전략 제안을 해줘."
)

url = "https://clovastudio.stream.ntruss.com/testapp/v1/tasks/fv49n0yw/chat-completions"
headers = {
    "Authorization": "Bearer <api-key>",  
    "X-NCP-CLOVASTUDIO-REQUEST-ID": "",       
    "Content-Type": "application/json; charset=utf-8"
}

data = {
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ],
    "temperature": 0.3,
    "topK": 0,
    "topP": 0.8,
    "repeatPenalty": 5.0,
    "maxTokens": 256,
    "includeAiFilters": True,
    "stopBefore": []
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

assistant_response = result.get("result", {}).get("message", {}).get("content", "")

output_data = {
    "product_id": product_data["product_id"],
    "assistant_response": assistant_response
}

with open(os.path.join(json_dir, f"{product_data['product_id']}_response.json"), "w", encoding="utf-8") as out_file:
    json.dump(output_data, out_file, ensure_ascii=False, indent=2)

print("저장 완료:")
pprint(output_data)