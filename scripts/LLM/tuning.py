# -*- coding: utf-8 -*-

import requests

class CreateTaskExecutor:
    def __init__(self, host, uri, api_key, request_id):
        self._host = host
        self._uri = uri
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, create_request):

        headers = {
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }
        result = requests.post(self._host + self._uri, json=create_request, headers=headers).json()
        return result

    def execute(self, create_request):
        res = self._send_request(create_request)
        if 'status' in res and res['status']['code'] == '20000':
            return res['result']
        else:
            return res


if __name__ == '__main__':
    completion_executor = CreateTaskExecutor(
        host='https://clovastudio.stream.ntruss.com',
        uri='/tuning/v2/tasks',
        api_key='Bearer <api-key>',
        request_id=''
    )

    request_data = {'name': 'generation_task',
                    'model': 'HCX-003',
                    'tuningType': 'PEFT',
                    'taskType': 'GENERATION',
                    'trainEpochs': '8',
                    'learningRate': '1e-5f',
                    'trainingDatasetBucket': 'recsys01storage',
                    'trainingDatasetFilePath': 'data_augment.csv',
                    'trainingDatasetAccessKey': 'ACCESS_KEY',
                    'trainingDatasetSecretKey': 'SECRET_KEY'
                    }
    response_text = completion_executor.execute(request_data)
    print("요청 데이터:", request_data)
    print("응답 결과:", response_text)
