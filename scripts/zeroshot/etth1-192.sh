if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
if [ ! -d "./logs/LongForecasting/transfer" ]; then
    mkdir ./logs/LongForecasting/transfer
fi
seq_len=96
model_name=ExoLLM
str='ETTh2'
if [ ! -d "./logs/LongForecasting/transfer/$str" ]; then
    mkdir ./logs/LongForecasting/transfer/$str
fi
dir=./logs/LongForecasting/transfer/$str
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2021
patch_len=8
stride=8

pred_len=192
loss='mae'
lr=1e-3
batch_size=256
e_layers=6
dropout=0.3
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
    --enc_in 7 \
    --e_layers $e_layers \
    --d_model 768 \
    --d_ff 32 \
    --dropout $dropout\
    --fc_dropout 0.3\
    --head_dropout 0\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --patience 10\
    --loss $loss\
    --train_epochs 40\
    --gpu 7\
    --itr 1 --batch_size $batch_size --learning_rate $lr >$dir/$model_id_name'_'$seq_len'_'$pred_len'_'$window_size'_'$patch_len'_'$stride'_'$dropout'_'$e_layers'_'$batch_size'_'$lr'_'$loss.log 

transfer_name=ETTh2
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 0 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --transfer_name $transfer_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --e_layers $e_layers \
    --d_model 768 \
    --d_ff 32 \
    --dropout $dropout\
    --patch_len $patch_len\
    --stride $stride\
    --des 'Exp' \
    --loss $loss\
    --gpu 7\
    --itr 1 --batch_size $batch_size --learning_rate $lr >$dir/$model_id_name'_transfer'$transfer_name'_'$seq_len'_'$pred_len.log
