#PBS -S /bin/bash
#PBS -lnodes=1
#PBS -qgpu
#PBS -lwalltime=08:00:00
module load python/3.5.0
python3 /home/lgpu0088/dl4nlp/gru_padded/eval.py  --save_path /home/lgpu0088/dl4nlp/gru_padded/models/ --model_path _last_model.pth  >> /home/lgpu0088/dl4nlp/gru_padded/models/final_acc.txt
