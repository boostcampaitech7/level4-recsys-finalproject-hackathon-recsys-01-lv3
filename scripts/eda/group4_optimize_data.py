import polars as pl
pl.Config.set_tbl_cols(100)

# 데이터 경로 지정
data_paths = ['../../data/Group_4.parquet', '../../data/Group_4_small.parquet']
output_paths = ['../../data/opt_data/Group_4_opt.parquet', '../../data/opt_data/Group_4_small_opt.parquet']

for data_path, output_path in zip(data_paths, output_paths):
    # 데이터 읽기
    df = pl.read_parquet(data_path)
    category_df = pl.read_parquet('../../data/category_table.parquet')
    brand_df = pl.read_parquet('../../data/brand_table.parquet')

    # 필요한 열 선택 및 캐스팅
    category_selected = category_df.select([
        'category_codes',
        'category_1_id',
        'category_2_id',
        'category_3_id',
        'category_4_id',
        'category_5_id'
    ]).with_columns(
        pl.col('category_1_id').cast(pl.UInt8),
        pl.col('category_2_id').cast(pl.UInt16),
        pl.col('category_3_id').cast(pl.UInt16),
        pl.col('category_4_id').cast(pl.UInt16),
        pl.col('category_5_id').cast(pl.UInt8)
    )

    # df3에서 필요한 열 선택 및 확인
    brand_selected = brand_df.select([
        'brand',
        'brand_id'
    ]).with_columns(
        pl.col('brand_id').cast(pl.UInt16)
    )

    # category_codes를 기준으로 병합
    merged1_df = df.join(category_selected, on='category_codes', how='left')

    # brand를 기준으로 병합
    merged2_df = merged1_df.join(brand_selected, on='brand', how='left')

    # 불필요한 열 제거
    all_df = merged2_df.drop(['category_codes', 'category_1', 'brand'])

    # 결과 출력
    print(all_df)

    # 결과 저장
    all_df.write_parquet(output_path)