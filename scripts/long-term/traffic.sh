if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=ExoLLM
str=traffic
if [ ! -d "./logs/LongForecasting/$str" ]; then
    mkdir ./logs/LongForecasting/$str
fi
dir=./logs/LongForecasting/$str
root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=traffic

random_seed=2021
loss=mse
dropout=0.3
bs=24
lr=1e-4
for pred_len in 96 192 336 720; do
    python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
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
        --e_layers 6 \
        --batch_size $bs \
        --learning_rate $lr \
        --loss $loss \
        --d_model 768 \
        --des 'Exp' \
        --patience 5 \
        --train_epochs 40 \
        --gpu 5 \
        --use_multi_gpu \
        --devices 0,1,2,3 \
        --itr 1 >$dir/$model_id_name'_'$seq_len'_'${pred_len}'_'${bs}'_'${lr}_${dropout}_${loss}.log 
done