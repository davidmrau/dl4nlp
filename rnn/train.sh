#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00
module load python/3.5.0
FOLDER=$HOME/dl4nlp/gru_padded/
python3 $FOLDER/train_gru.py --test $FOLDER/test.txt --train $FOLDER/train.txt  --save_path $FOLDER/models/ >> $FOLDER/models/log.txt
