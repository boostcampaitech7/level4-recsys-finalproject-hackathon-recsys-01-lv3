# PROMO - Prescriptive AI for Promotion

## 🗂️ 목차
- [프로젝트 소개](#프로젝트-소개)
- [팀 소개](#팀-소개)
- [개발 환경 및 기술 스택](#개발-환경-및-기술-스택)
- [디렉토리 구조](#디렉토리-구조)
- [전체 모델 아키텍처](#전체-모델-아키텍처)
<br/>

## 💡 프로젝트 소개
"**PROMO**"는 프로모션 광고 비용을 절감하고 매출을 높일 수 있도록, 프로모션 시 구매 가능성이 높은 유저군을 선별하여 최적 할인율과 유저 수를 동적으로 추천해주는 AI 솔루션입니다. 사용자는 원하는 상품 정보를 선택하고, 추천된 할인율과 프로모션 대상 인원을 조정해가면서, 예상 매출과 최적값 대비 차이를 실시간으로 확인할 수 있습니다. 미등록 상품의 경우, 상품 정보를 입력하면 유사 상품 추천을 통해 프로모션 전략을 확인할 수 있습니다.
<p align="center"><img src="https://github.com/user-attachments/assets/34d0bb78-e59b-4c9a-a198-78439b72fa6d"></p>

### 프로모션 시 구매 가능성이 높은 유저군을 선별합니다.
추천시스템 모델 MBGCN을 통해 다양한 이벤트 정보를 활용하여 구매 가능성이 높은 유저군을 선별합니다.
### 최적 할인율과 유저수를 동적으로 추천합니다.
강화 학습 모델 HRL을 통해 추천 모델이 선정한 유저 후보군과 상품 정보를 바탕으로 할인율과 유저수를 최적화합니다.
### 광고 비용을 절감하면서 매출을 높일 수 있습니다.
PROMO를 통해 효과적인 마케팅 전략을 수립하고, 타겟 고객에게 적절한 혜택을 제공하여 구매 전환율을 높일 수 있습니다.

<br/>

## 🍚 팀 소개
네이버 부스트캠프 AI Tech 7기 Level 4 RecSys 1조 **오곡밥** 입니다.  
### 오곡밥의 의미
오늘 곡소리 나도록 열공해서 밥값은 하자 🙂
### 팀원 소개
|문원찬|안규리|오소영|오준혁|윤건욱|황진욱|
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/a29cbbd9-0cde-495a-bd7e-90f20759f3d1" width="100"/> | <img src="https://github.com/user-attachments/assets/c619ed82-03f3-4d48-9bba-dd60408879f9" width="100"/> | <img src="https://github.com/user-attachments/assets/1b0e54e6-57dc-4c19-97f5-69b7e6f3a9b4" width="100"/> | <img src="https://github.com/user-attachments/assets/67d19373-8cac-4676-bde1-b0637921cf7f" width="100"/> | <img src="https://github.com/user-attachments/assets/f91dd46e-9f1a-42e7-a939-db13692f4098" width="100"/> | <img src="https://github.com/user-attachments/assets/69bbb039-752e-4448-bcaa-b8a65015b778" width="100"/> |
| [![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/WonchanMoon)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/notmandarin)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/irrso)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/ojunhyuk99)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/YoonGeonWook)|[![GitHub Badge](https://img.shields.io/badge/github-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/hw01931)|

<br/>

## ⚙️ 개발 환경 및 기술 스택
### 개발 환경
- OS: Linux (5.4.0-99-generic, x86_64)  
- GPU: Tesla V100-SXM2-32GB (CUDA Version: 12.2)  
- CPU : Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz, 8 Cores  

### 기술 스택
- 프로그래밍 언어: <img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=python&logoColor=white"/>
- 데이터 분석 및 처리: <img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white"/> <img src="https://img.shields.io/badge/Polars-CD792C.svg?style=flat-square&logo=polars&logoColor=white"/> <img src="https://img.shields.io/badge/PyArrow-D22128.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=scipy&logoColor=white"/> 
- 데이터 시각화: <img src="https://img.shields.io/badge/Matplotlib-013243.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white"/> <img src="https://img.shields.io/badge/seaborn-565C89.svg?style=flat-square&logoColor=white"/>
- 머신러닝 및 딥러닝: <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white"/> <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=flat-square&logo=scikitlearn&logoColor=white"/> 
- 강화학습: <img src="https://img.shields.io/badge/Stable--Baselines3-ee4c2c.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/Gymnasium-000000.svg?style=flat-square&logoColor=white"/> <img src="https://img.shields.io/badge/Shimmy-000000.svg?style=flat-square&logoColor=white"/>
- 웹 스크래핑: <img src="https://img.shields.io/badge/BeautifulSoup4-1668F5.svg?style=flat-square&logoColor=white"/>
- 개발 도구: <img src="https://img.shields.io/badge/Poetry-60A5FA.svg?style=flat-square&logo=poetry&logoColor=white"/> <img src="https://img.shields.io/badge/Commitizen-000000.svg?style=flat-square&logoColor=white"/>
<br/>

## 📂 디렉토리 구조
```
# level4-recsys-finalproject-hackathon-recsys-01-lv3/
│
├── .github/
│   └── ISSUE_TEMPLATE
│       ├── bug_report.md
│       ├── feature_request.md
│       ├── PULL_REQUEST_TEMPLATE.md
│       └── .keep
│
├── __pycache__/
│   └── cz_customize.cpython-311.pyc
│
├── config/
│   └── rl.yaml
│
├── experiments/
│   ├── optuna_tuning_mbgcn.py
│   └── optuna_tuning_mf.py
│
├── scripts/
│   ├── LLM/
│   │   ├── after_tuning.py
│   │   ├── embedding_results.py
│   │   ├── example.py
│   │   ├── inference.py
│   │   ├── post_llm.py
│   │   └── tuning.py
│   ├── papago_translation/
│   │   ├── papago_translation.log
│   │   └── papago_translation.py
│   ├── hrl_dqn_td3.sh
│   ├── main_recysy.py
│   ├── main_rl.py
│   ├── mbgcn_grid.sh
│   └── mf_grid.sh
│
├── service/
│   ├── backend/
│   └── frontend/
│
└── src/
│   ├── WebScrapping/
│   │   └── crawling.py
│   ├── agent/
│   │   └── DQNTD3Agent.py
│   ├── data/
│   │   ├── preprocess_for_rl
│   │   │   ├── cal_elasticity.py
│   │   │   ├── data_preprocess.py
│   │   │   └── environment.py
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── item_FE.py
│   │   ├── load_data.py
│   │   ├── preparation.py
│   │   ├── preprocessor.py
│   │   └── user_FE.py
│   ├── models/
│   │   ├── mbgcn.py
│   │   ├── mbgcn_multi_layer.py
│   │   └── rl_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── replay_buffer.py
│   │   └── utility.py
│
├── cz.yaml
├── .gitignore
├── poetry.lock
└── pyproject.toml
```
<br/>

## 🏛️ 전체 모델 아키텍처
![image](https://github.com/user-attachments/assets/f155a9f6-aaa3-4018-a4ea-34098120d456)

### 추천 시스템 아키텍처
![image](https://github.com/user-attachments/assets/6037ac93-16ee-443e-9a47-2bc61e81d173)

### 강화 학습 아키텍처
![image](https://github.com/user-attachments/assets/8e05422b-fdf9-4b37-87d3-ab4b08bf28db)
