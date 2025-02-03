export CUDA_VISIBLE_DEVICES=0

model_name=CD_Ister

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 64 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --d_model 32 \
  --patience 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 64 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --d_model 32 \
  --patience 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 64 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 35 \
  --d_model 32 \
  --patience 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --layers 1 \
  --chan_in 7 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 64 \
  --learning_rate 0.0003 \
  --lradj cosine \
  --train_epochs 50 \
  --d_model 32 \
  --patience 10
