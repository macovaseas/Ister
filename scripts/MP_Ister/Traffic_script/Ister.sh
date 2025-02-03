export CUDA_VISIBLE_DEVICES=0

model_name=MP_Ister

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --layers 1 \
  --chan_in 862 \
  --d_model 512 \
  --top_k 2 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 3


python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_96_192 \
 --model $model_name \
 --data custom \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 192 \
 --layers 1 \
 --chan_in 862 \
 --d_model 512 \
 --top_k 2 \
 --des 'Exp' \
 --itr 1 \
 --batch_size 4 \
 --learning_rate 0.001 \
 --train_epochs 10 \
 --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_96_336 \
 --model $model_name \
 --data custom \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 336 \
 --layers 1 \
 --chan_in 862 \
 --d_model 512 \
 --top_k 2 \
 --des 'Exp' \
 --itr 1 \
 --learning_rate 0.001 \
 --batch_size 4 \
 --train_epochs 10 \
 --patience 3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 1 \
 --root_path dataset/traffic/ \
 --data_path traffic.csv \
 --model_id traffic_96_720 \
 --model $model_name \
 --data custom \
 --seq_len 96 \
 --label_len 48 \
 --pred_len 720 \
 --layers 1 \
 --chan_in 862 \
 --d_model 512 \
 --top_k 2 \
 --des 'Exp' \
 --itr 1 \
 --batch_size 4 \
 --learning_rate 0.001 \
 --train_epochs 10 \
 --patience 3