git init
dvc init
dvc remote add -d localremote ../local_remote
dvc run -n prepare \
        -d prepare.py -d fake_data.csv \
        -o X.csv -o y.csv \
        python prepare.py

dvc run -n train \
        -d train.py -d X.csv -d y.csv \
        -p train.lr_C \
        python train.py