#!/bin/bash

# mf_grid.sh
# 다양한 (lr, l2) 조합으로 train_mf_tmall.py 실행 -> epoch=400, patience=40

embedding_size=64
epoch=400
patience=10
batch_size=4096
root_dir="src/data/MBGCN/Tmall"

# lr_list=('3e-4' '1e-4')
# l2_list=('1e-3' '1e-4' '1e-5')
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
        python src/train_mf_tmall.py \
            --epoch ${epoch} \
            --patience ${patience} \
            --lr ${lr} \
            --l2_reg ${l2} \
            --batch_size ${batch_size} \
            --embedding_size ${embedding_size} \
            --save_path ${save_path} \
            --root_dir ${root_dir}

        echo "Done => saved to ${save_path}"
        echo
    done
done
