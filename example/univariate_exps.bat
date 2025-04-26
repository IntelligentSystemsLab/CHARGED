@echo off

set cities=AMS SZH SPO JHB LOA MEL
set models=arima ar segrnn frets convtimenet multipatchformer
set folds=1 2 3 4 5 6

for %%m in (%models%) do (
    for %%c in (%cities%) do (
        for %%f in (%folds%) do (
                python univariate_prediction.py --city %%c --model %%m --fold %%f|| (
                    echo Error occurred on city %%c, model %%m, fold %%f
                    pause
                    exit /b 1
                )
            )
        )
)
echo All experiments completed.
