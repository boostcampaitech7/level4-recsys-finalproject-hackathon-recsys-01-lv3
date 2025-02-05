#!/bin/bash
# mbgcn_grid.sh
# MBGCN 모델에 대해 lr과 l2_reg에 대해 grid search하는 스크립트

embedding_size=64
epoch=400
patience=40
train_batch_size=4096
valid_batch_size=16384
device="cuda"
data_path="src/data/MBGCN/data"
dataset_name="final"
relations="cart,view"
save_base_dir="src/data/MBGCN/data/final/trained_models"

# MBGCN 고정 인자들
lamb=0.5
node_dropout=0.2
message_dropout=0.2
mgnn_weight="[1.0, 1.0]"

# lr과 l2_reg의 grid search 
# lr_list=('1e-3', '3e-4', '1e-4')
lr_list=('1e-3')
l2_list=('1e-4')

for lr in "${lr_list[@]}"
do
    for l2 in "${l2_list[@]}"
    do
        exp_name="MBGCN_lr${lr}_L2${l2}_dim${embedding_size}"
        save_path="${save_base_dir}/${exp_name}"
        
        # 사전 학습된 MF 모델 디렉토리 (model.pkl 파일 포함)
        pretrain_dir=""
        
        echo "Running MBGCN with lr=${lr}, l2_reg=${l2}, emb=${embedding_size}"
        echo "Using pretrain model from: ${pretrain_dir}/model.pkl"
        
        python main.py \
            --model MBGCN \
            --lr ${lr} \
            --l2_reg ${l2} \
            --train_batch_size ${train_batch_size} \
            --valid_batch_size ${valid_batch_size} \
            --epoch ${epoch} \
            --patience ${patience} \
            --embedding_size ${embedding_size} \
            --mgnn_weight "${mgnn_weight}" \
            --lamb ${lamb} \
            --node_dropout ${node_dropout} \
            --message_dropout ${message_dropout} \
            --data_path ${data_path} \
            --dataset_name ${dataset_name} \
            --relations ${relations} \
            --save_path ${save_path} \
            --device ${device}

        echo "Done => saved to ${save_path}"
        echo
    done
done
