#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00
module load python/3.5.0
FOLDER=$HOME/dl4nlp/gru_padded/
python3 $FOLDER/train_gru_emb.py --test $FOLDER/test.txt   --train $FOLDER/train.txt --save_path $FOLDER/embedding_models/ >> $FOLDER/embedding_models/log.txt
