import http.client
import json
import os
import re
import time

embedding_folder = '/data/ephemeral/home/Hackathon/data/embedding'
output_file = '/data/ephemeral/home/Hackathon/data/embedding_results.json'


class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id
        self.suc_count = 0

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/api-tools/embedding/v2', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result
    
    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            self.suc_count += 1
            return res['result']['embedding']
        else:
            return 'Error'


def preprocess_text(text):
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[^\w\s:\.,\-/]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='clovastudio.stream.ntruss.com',
        api_key='Bearer <api-key>',
        request_id=''
    )

    embedding_results = {}

    batch_size = 30

    all_files = [f for f in os.listdir(embedding_folder) if f.endswith('.json')]
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i + batch_size]
        print(f"{i + 1}번째 배치 처리 중 (총 {len(batch_files)}개 파일)")

        for filename in batch_files:
            try:
                product_id = int(filename.split('_')[1].split('.')[0])
                file_path = os.path.join(embedding_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                raw_text = data.get("text", "")
                cleaned_text = preprocess_text(raw_text)

                request_data = {"text": cleaned_text}

                response_text = completion_executor.execute(request_data)

                embedding_results[product_id] = response_text

            except Exception as e:
                print(f"파일 {filename} 처리 중 오류 발생: {e}")

        print(f"{i + len(batch_files)}개 파일 처리 완료. 30초 대기 중...")
        time.sleep(30)

    with open(output_file, 'w', encoding='utf-8') as output:
        json.dump(embedding_results, output, indent=2, ensure_ascii=False)

    print(f"임베딩 결과가 {output_file}에 저장되었습니다.")