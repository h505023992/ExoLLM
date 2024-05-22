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
for data_path_name in "PJM" "NP" "BE" "FR" "DE";do # "PJM" "NP" "BE" "FR" "DE" 
    model_id_name=${data_path_name}
    data_name=${data_path_name}
    str=${data_path_name}
    if [ ! -d "./logs/LongForecasting/$str" ]; then
        mkdir ./logs/LongForecasting/$str
    fi
    dir=./logs/LongForecasting/$str

    dropout=0.3
    bs=256
    lr=1e-4
    e_layers=6
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
