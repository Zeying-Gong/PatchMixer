if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchMixer

root_path_name=./dataset/weather
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 1 \
      --d_model 256 \
      --dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8 \
      --des 'Exp' \
      --train_epochs 200\
      --patience 10\
      --loss_flag 2\
      --itr 1 --batch_size 1024 --learning_rate 0.001 >logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log
done