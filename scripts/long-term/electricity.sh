if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=ExoLLM
str=electricity
if [ ! -d "./logs/LongForecasting/$str" ]; then
    mkdir ./logs/LongForecasting/$str
fi
dir=./logs/LongForecasting/$str
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=electricity
random_seed=2021
bs=12
lr=1e-4
loss=mae
dropout=0.3
for lr in 1e-4 5e-4 1e-3 5e-3; do
for pred_len in 720 336 192 96; do
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
        --enc_in 321 \
        --dropout $dropout \
        --e_layers 6 \
        --batch_size 3852 \
        --learning_rate $lr \
        --loss $loss \
        --d_model 768 \
        --des 'Exp' \
        --patience 5 \
        --train_epochs 30 \
        --gpu 0 \
        --itr 1 >$dir/$model_id_name'_'$seq_len'_'${pred_len}'_'${bs}'_'${lr}_${dropout}_${loss}.log 
done
done