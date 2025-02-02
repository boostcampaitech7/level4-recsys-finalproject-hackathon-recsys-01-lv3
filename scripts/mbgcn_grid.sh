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
dataset_name="Tmall"
relations="buy,cart,click,collect"
save_base_dir="Hackathon/src/data/MBGCN/pretrained_model"

# MBGCN 고정 인자들
lamb=0.5
node_dropout=0.2
message_dropout=0.2
alpha_mode="global"
alpha_learning="--alpha_learning"   # flag: 존재하면 활성화
item_alpha="--item_alpha"             # flag: 존재하면 활성화
item_cf_mode="bigmat"

# lr과 l2_reg의 grid search 
lr_list=('1e-3' '3e-4' '1e-4')
l2_list=('1e-3' '1e-4' '1e-5')

for lr in "${lr_list[@]}"
do
    for l2 in "${l2_list[@]}"
    do
        exp_name="MBGCN_lr${lr}_L2${l2}_dim${embedding_size}"
        save_path="${save_base_dir}/${exp_name}"
        
        # 사전 학습된 MF 임베딩 경로: 
        # 경로 구조: Hackathon/src/data/MBGCN/Tmall/pretrained_mf_runs/MF_lr<lr>_L2<l2>_dim<embedding_size>/
        pretrain_dir="Hackathon/src/data/MBGCN/Tmall/pretrained_mf_runs/MF_lr${lr}_L2${l2}_dim${embedding_size}"
        pretrain_user_emb="${pretrain_dir}/user_emb.npy"
        pretrain_item_emb="${pretrain_dir}/item_emb.npy"
        
        echo "Running MBGCN with lr=${lr}, l2_reg=${l2}, emb=${embedding_size}, num_layers=${num_layers}"
        echo "Using pretrain embeddings from:"
        echo "   user: ${pretrain_user_emb}"
        echo "   item: ${pretrain_item_emb}"
        
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
            --alpha_mode ${alpha_mode} \
            ${alpha_learning} \
            ${item_alpha} \
            --item_cf_mode ${item_cf_mode} \
            --use_pretrain \
            --pretrain_user_emb ${pretrain_user_emb} \
            --pretrain_item_emb ${pretrain_item_emb} \
            --data_path ${data_path} \
            --dataset_name ${dataset_name} \
            --relations ${relations} \
            --save_path ${save_path} \
            --device ${device}

        echo "Done => saved to ${save_path}"
        echo
    done
done