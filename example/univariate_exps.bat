@echo off

set cities=SPO AMS JHB LOA MEL SZH
set models=ar lo arima fcnn lstm
set folds=1 2 3 4 5 6

for %%c in (%cities%) do (
    for %%m in (%models%) do (
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
