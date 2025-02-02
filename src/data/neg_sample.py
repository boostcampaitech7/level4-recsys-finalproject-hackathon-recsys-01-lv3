#!/usr/bin/env python
import os
import argparse
import random

def generate_negative_samples(train_file, num_users, num_items, neg_sample_size=1):
    """
    Args:
        - train_file: 각 행이 "user item" 형태의 positive sample을 담고 있음
        - num_users: 총 사용자 수
        - num_items: 총 아이템 수
        - neg_sample_size: 각 positive sample 당 생성할 negative sample 수
    Returns:
        - 각 행이 "user pos_item neg_item" 형태인 문자열 리스트
    """
    # 사용자별로 이미 본 아이템 집합 생성
    user_positive = {}
    positive_samples = [] # (user, pos_item) 튜플 리스트
    with open(train_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user, item = int(parts[0]), int(parts[1])
            positive_samples.append((user, item))
            if user not in user_positive:
                user_positive[user] = set()
            user_positive[user].add(item)
    sample_lines = []
    for (user, pos_item) in positive_samples:
        negatives = []
        # 이미 본 아이템을 제외한 임의의 negative item을 선택
        while len(negatives) < neg_sample_size:
            neg_item = random.randint(0, num_items - 1)
            if neg_item not in user_positive.get(user, set()):
                negatives.append(neg_item)
        for neg in negatives:
            sample_lines.append(f"{user} {pos_item} {neg}\n")
    return sample_lines

def main():
    parser = argparse.ArgumentParser(description="Generate negative samples for training")
    parser.add_argument("--path", type=str, required=True,
                        help="Root path to the dataset folder, e.g., 'src/data/MBGCN/'")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name folder, e.g., 'Tmall'")
    parser.add_argument("--neg_sample_size", type=int, default=1,
                        help="Number of negative samples per positive sample")
    args = parser.parse_args()
    
    # 파일 경로 설정
    train_file = os.path.join(args.path, args.dataset_name, "train.txt")
    size_file = os.path.join(args.path, args.dataset_name, "data_size.txt")
    
    if not os.path.exists(train_file):
        print(f"Error: Train file '{train_file}' not found!")
        return
    if not os.path.exists(size_file):
        print(f"Error: Data size file '{size_file}' not found!")
        return
    
    # data_size.txt 파일에서 사용자 수와 아이템 수 읽기 (첫 번째 줄: "num_users num_items")
    with open(size_file, 'r') as f:
        line = f.readline().strip()
        try:
            num_users_str, num_items_str = line.split()
            num_users = int(num_users_str)
            num_items = int(num_items_str)
        except Exception as e:
            print("Error parsing data_size.txt:", e)
            return
    print(f"Found {num_users} users and {num_items} items in the dataset.")
    
# 부정 샘플 생성
    sample_lines = generate_negative_samples(train_file, num_users, num_items, args.neg_sample_size)
    print(f"Generated {len(sample_lines)} training samples (with negative sampling).")
    
    # sample_file 디렉토리 생성 (없으면 생성)
    sample_dir = os.path.join(args.path, args.dataset_name, "sample_file")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"Created directory: {sample_dir}")
    
    # sample_0.txt 파일에 저장 (필요시 epoch 인덱스를 조정 가능)
    sample_file = os.path.join(sample_dir, "sample_0.txt")
    with open(sample_file, 'w') as f:
        f.writelines(sample_lines)
    print(f"Negative samples saved to: {sample_file}")

if __name__ == "__main__":
    main()