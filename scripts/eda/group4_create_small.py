import polars as pl
import random

# 데이터 로드
data_path = '../../data/Group_4.parquet'
df = pl.read_parquet(data_path)

# 결과 저장할 리스트
samples = []

# 카테고리별로 처리
categories = df['category_1'].unique()
for category in categories:
    # 해당 카테고리 데이터 필터링
    category_df = df.filter(pl.col('category_1') == category)
    
    # 영어 브랜드
    english_brands = category_df.filter(pl.col('brand').str.contains(r'^[a-zA-Z\s]+$', literal=False))
    
    # 카자흐스어 또는 러시아어 브랜드
    kazakh_russian_brands = category_df.filter(pl.col('brand').str.contains(r'[а-яА-ЯёЁ]', literal=False))
    
    # 각각 랜덤으로 25개 샘플링
    english_sample = english_brands.sample(n=25, seed=42) if english_brands.shape[0] >= 5 else english_brands
    kazakh_russian_sample = kazakh_russian_brands.sample(n=25, seed=42) if kazakh_russian_brands.shape[0] >= 5 else kazakh_russian_brands
    
    # 합치기
    combined_sample = pl.concat([english_sample, kazakh_russian_sample])
    
    samples.append(combined_sample)

# 결과를 하나의 데이터프레임으로 결합
result = pl.concat(samples)

# 결과 출력
print(result)

# 파일 생성
output_path = '../../data/Group_4_small.parquet'
result.write_parquet(output_path)