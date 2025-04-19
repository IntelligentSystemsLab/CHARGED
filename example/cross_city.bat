@echo off

set max_stations_choice=25 50 100 150 200

for %%c in (%max_stations_choice%) do (
    python knowledge_transfer.py --city AMS+JHB+LOA+MEL+SPO+SZH --eval_city SZH --auxiliary all --model frets --pred_type city --global_epoch 100 --deploy_epoch 20 --max_stations %%c|| (
                    echo Error occurred on max_stations %%c
                    pause
                    exit /b 1
                )
)
echo All experiments completed.
