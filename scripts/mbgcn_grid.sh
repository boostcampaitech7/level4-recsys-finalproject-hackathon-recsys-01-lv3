#!/bin/bash
# mbgcn_grid.sh
# MBGCN 모델에 대해 lr과 l2_reg에 대해 grid search하는 스크립트

embedding_size=64
num_layers=2
epoch=400
patience=40
batch_size=4096
device="cuda"
data_path="Hackathon/src/data/MBGCN"
dataset_name="data/smartphones"
relations="purchase,cart,view"
save_base_dir="Hackathon/src/data/MBGCN/data/smartphones/pretrained_models"

# MBGCN 고정 인자들
lamb=0.5
node_dropout=0.2
message_dropout=0.2

# lr과 l2_reg의 grid search 
lr_list=('1e-3')
l2_list=('1e-3')

for lr in "${lr_list[@]}"
do
    for l2 in "${l2_list[@]}"
    do
        exp_name="MBGCN_lr${lr}_L2${l2}_dim${embedding_size}"
        save_path="${save_base_dir}/${exp_name}"
        
        # 사전 학습된 MF 모델 디렉토리 (model.pkl 파일 포함)
        pretrain_dir="Hackathon/src/data/MBGCN/Tmall/pretrained_mf_runs/MF_lr${lr}_L2${l2}_dim${embedding_size}"
        
        echo "Running MBGCN with lr=${lr}, l2_reg=${l2}, emb=${embedding_size}, num_layers=${num_layers}"
        echo "Using pretrain model from: ${pretrain_dir}/model.pkl"
        
        python main.py \
            --model MBGCN \
            --lr ${lr} \
            --l2_reg ${l2} \
            --batch_size ${batch_size} \
            --epoch ${epoch} \
            --patience ${patience} \
            --embedding_size ${embedding_size} \
            --num_layers ${num_layers} \
            --lamb ${lamb} \
            --node_dropout ${node_dropout} \
            --message_dropout ${message_dropout} \
            --use_pretrain \
            --pretrain_path ${pretrain_dir} \
            --pretrain_frozen False \
            --data_path ${data_path} \
            --dataset_name ${dataset_name} \
            --relations ${relations} \
            --save_path ${save_path} \
            --device ${device}

        echo "Done => saved to ${save_path}"
        echo
    done
done
