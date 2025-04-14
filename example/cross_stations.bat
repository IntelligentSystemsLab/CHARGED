@echo off

python knowledge_transfer.py --city SZH --max_stations 200 --eval_percentage 20 --auxiliary all --model frets --pred_type station --global_epoch 100 --deploy_epoch 20

echo All experiments completed.
