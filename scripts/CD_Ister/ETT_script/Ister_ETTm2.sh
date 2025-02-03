export CUDA_VISIBLE_DEVICES=0

model_name=CD_Ister

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --factor 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 20 \
  --d_model 128 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --layers 1 \
  --factor 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 20 \
  --d_model 128 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --layers 1 \
  --factor 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 20 \
  --d_model 32 \

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --layers 1 \
  --factor 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 512 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 20 \
  --d_model 128 \
