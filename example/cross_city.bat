@echo off

python knowledge_transfer.py --city AMS+JHB+LOA+MEL+SPO+SZH --eval_city SZH --auxiliary all --model frets --pred_type city --global_epoch 100 --deploy_epoch 20

echo All experiments completed.
