import polars as pl
import os
import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

FILE_PATH = os.path.join(config["data"]["group"], "Group_3.parquet")
PP_FILE_PATH = os.path.join(config["data"]["group"], "PP_Group_3.parquet")

print(f"file_path : {FILE_PATH}")

df = pl.read_parquet(FILE_PATH)

print("\n결측치 제거")
df = df.drop_nulls()


print("\n파일 저장")
df.write_parquet(PP_FILE_PATH)