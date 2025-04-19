@echo off

python knowledge_transfer.py --city AMS+JHB+LOA+MEL+SPO+SZH --max_stations 30 --eval_percentage 20 --auxiliary all --model frets --pred_type station --global_epoch 100 --deploy_epoch 20 --output_path ./result/federated/all/

echo All experiments completed.
