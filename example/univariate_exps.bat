@echo off

set cities=AMS SZH SPO JHB LOA MEL
@REM set cities=AMS
@REM set models=arima ar lo fcnn lstm segrnn frets moderntcn multipatchformer
set models=moderntcn multipatchformer
@REM set folds=1 2 3 4 5 6
set folds=6 5 4 3 2 1

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
