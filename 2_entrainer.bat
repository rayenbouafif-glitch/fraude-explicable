@echo off
REM Utilise data\fraudTrain.csv + data\fraudTest.csv s'ils existent, sinon le jeu synthetique
python train.py %*
pause
