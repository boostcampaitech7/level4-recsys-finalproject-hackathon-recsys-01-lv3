# -*- coding: utf-8 -*-
import requests
from pprint import pprint

class FindTaskExecutor:
    def __init__(self, host, uri, api_key, request_id):
        self._host = host
        self._uri = uri
        self._api_key = api_key
        self._request_id = request_id

    def _send_request(self, task_id):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'Authorization': self._api_key,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        result = requests.get(self._host + self._uri + task_id, headers=headers).json()
        return result

    def execute(self, taskId):
        res = self._send_request(taskId)
        if 'status' in res and res['status']['code'] == '20000':
            return res['result']
        else:
            return res


if __name__ == '__main__':
    completion_executor = FindTaskExecutor(
        host='https://clovastudio.stream.ntruss.com',
        uri='/tuning/v2/tasks/',
        api_key='Bearer <api-key>',
        request_id=''
    )

    taskId = 'fv49n0yw'
    response_text = completion_executor.execute(taskId)
    pprint(taskId)
    pprint(response_text)
