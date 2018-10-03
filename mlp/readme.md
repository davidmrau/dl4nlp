# Pipeline
-


### data pre-processing

```
usage: preprocessing_mlp.py [-h] --X_train X_TRAIN --y_train Y_TRAIN --X_test
                            X_TEST --y_test Y_TEST
                            [--min_char_freq MIN_CHAR_FREQ]
                            [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --X_train X_TRAIN     Path to the training input data
  --y_train Y_TRAIN     Path to the preprocessed target train data
  --X_test X_TEST       Path to the test input data
  --y_test Y_TEST       Path to the target test data
  --min_char_freq MIN_CHAR_FREQ
                        use only chars with frequency >= min_char_freq
  --save_path SAVE_PATH
                        Output path for preprocessed training data: "train.p",
                        preprocessed test data:"test.p" and the index2lang
                        dicttionary: "index2lang.p"
```

### train model

```
usage: train.py [-h] --test TEST --train TRAIN [--num_hidden NUM_HIDDEN]
                [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --test TEST           Path to the preprocessed test data
  --train TRAIN         Path to the preprocessed train data
  --num_hidden NUM_HIDDEN
                        Number of hidden units
  --batch_size BATCH_SIZE
                        Number of examples to process in a batch
  --epochs EPOCHS       Number of training epochs
  --save_path SAVE_PATH
                        Output path for accuracy: "acc.p", evaluation
                        loss:"eval_loss.p", loss: "loss.p" and the model:
                        "model.pth"
```

### evaluate

```
usage: eval.py [-h] --test TEST --index2lang INDEX2LANG --model MODEL
               [--num_hidden NUM_HIDDEN]

optional arguments:
  -h, --help            show this help message and exit
  --test TEST           Path to the preprocessed test data
  --index2lang INDEX2LANG
                        Path index2lang dict
  --model MODEL         Path to model
  --num_hidden NUM_HIDDEN
                        Number of hidden units

```