if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=168
model_name=ExoLLM

root_path_name=./dataset/

random_seed=2021
pred_len=24
loss='mse'
for lr in 1e-4 3e-4 5e-5 8e-4 1e-3 3e-3 5e-3 8e-3;do
    for bs in 32 64 128 256 512;do
        for dropout in 0.1 0.2 0.3 0.4;do
            for e_layers in 1 2 3 4 5 6;do
                for data_path_name in "PJM" "NP" "BE" "FR" "DE";do # "PJM" "NP" "BE" "FR" "DE" 
                    model_id_name=${data_path_name}
                    data_name=${data_path_name}
                    str=${data_path_name}
                    if [ ! -d "./logs/LongForecasting/$str" ]; then
                        mkdir ./logs/LongForecasting/$str
                    fi
                    dir=./logs/LongForecasting/$str

                    python -u run_longExp.py \
                        --random_seed $random_seed \
                        --is_training 1 \
                        --patch_len 24 \
                        --root_path $root_path_name \
                        --data_path $data_path_name \
                        --model_id $model_id_name'_'$seq_len'_'$pred_len \
                        --model $model_name \
                        --data $data_name \
                        --features M \
                        --seq_len $seq_len \
                        --pred_len $pred_len \
                        --enc_in 862 \
                        --dropout $dropout \
                        --e_layers $e_layers \
                        --batch_size $bs \
                        --learning_rate $lr \
                        --loss $loss \
                        --d_model 768 \
                        --des 'Exp' \
                        --patience 5 \
                        --train_epochs 40 \
                        --gpu 4 \
                        --itr 1 >$dir/$model_id_name'_'$seq_len'_'${pred_len}'_'${bs}'_'${lr}_${dropout}_${e_layers}_${loss}.log 
                done
            done
        done
    done
done