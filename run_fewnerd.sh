
datasets_target=('FEW-NERD-INTRA')
n_way_k_shots=('5_1')

for _dataset_target in "${datasets_target[@]}"
   do
              python3 main.py --mode use_type_name \
                      --train True \
                      --dataset_source $_dataset_target \
                      --dataset_target $_dataset_target \
                      --n_way_k_shot '5_1' \
                      --seed 42 \
                      --test_stage1_only False\
                      --test_stage2_only False \
                      --filter True

              for _n_way_k_shot in "${n_way_k_shots[@]}"
                   do
                        python3 main.py --mode use_type_name \
                                        --train False \
                                        --dataset_source $_dataset_target \
                                        --dataset_target $_dataset_target \
                                        --n_way_k_shot $_n_way_k_shot \
                                        --seed 42 \
                                        --test_stage1_only False\
                                        --test_stage2_only False \
                                        --filter True
                   done
   done

