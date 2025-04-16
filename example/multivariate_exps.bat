@echo off

set cities=AMS SZH SPO JHB LOA MEL
set auxiliary=all e_price s_price temp precip visibility

for %%c in (%cities%) do (
    for %%a in (%auxiliary%) do (
            python multivariate_prediction.py --city %%c --model frets --output_path ./result/multivariate/ --auxiliary %%a|| (
                echo Error occurred on city %%c, auxiliary %%a
                pause
                exit /b 1
            )
        )
)
echo All experiments completed.
