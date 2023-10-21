
datasets_target=('CONLL2003')
k_shots=('1' '5')

python3 main.py --mode use_type_name \
                      --train True \
                      --dataset_source Ontonotes \
                      --dataset_target CONLL2003 \
                      --k_shot 1 \
                      --seed 42 \
                      --test_stage1_only False\
                      --test_stage2_only False \
                      --filter True

for _dataset_target in "${datasets_target[@]}"
    do
         for _k_shot in "${k_shots[@]}"
             do
                  python3 main.py --mode use_type_name \
                                  --train False \
                                  --dataset_source Ontonotes \
                                  --dataset_target $_dataset_target \
                                  --k_shot $_k_shot \
                                  --seed 42 \
                                  --test_stage1_only False\
                                  --test_stage2_only False \
                                  --filter True
             done
    done

