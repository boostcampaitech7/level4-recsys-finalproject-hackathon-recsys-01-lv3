#!/bin/bash

# mf_grid.sh
# 다양한 (lr, l2_reg) 조합으로 MF 모델 실험

embedding_size=64
epoch=400
patience=10
batch_size=4096
data_path="src/data/MBGCN"
dataset_name="Tmall"

lr_list=('1e-4')
l2_list=('1e-5')

save_base_dir="src/data/MBGCN/Tmall/pretrained_mf_runs"
mkdir -p ${save_base_dir}

for lr in ${lr_list[@]}
do
    for l2 in ${l2_list[@]}
    do
        name="MF_lr${lr}_L2${l2}_dim${embedding_size}"
        save_path="${save_base_dir}/${name}"

        echo "Running MF with lr=${lr}, l2_reg=${l2}, emb=${embedding_size}, epoch=${epoch}, patience=${patience}"
        python main.py \
            --model MF \
            --epoch ${epoch} \
            --patience ${patience} \
            --lr ${lr} \
            --l2_reg ${l2} \
            --batch_size ${batch_size} \
            --embedding_size ${embedding_size} \
            --save_path ${save_path} \
            --data_path ${data_path} \
            --dataset_name ${dataset_name}

        echo "Done => saved to ${save_path}"
        echo
    done
done
