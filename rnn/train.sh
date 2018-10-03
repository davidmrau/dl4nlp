#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00
module load python/3.5.0
FOLDER=$HOME/dl4nlp/rnn/
python3 $FOLDER/train_gru.py --test $FOLDER/data/test.txt --train $FOLDER/data/train.txt  --save_path $FOLDER/models_new/ >> $FOLDER/models_new/log.txt
