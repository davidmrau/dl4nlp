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
usage: train_gru.py [-h] --test TEST --train TRAIN
                    [--gru_num_hidden GRU_NUM_HIDDEN]
                    [--gru_num_layers GRU_NUM_LAYERS]
                    [--dropout_keep_prob DROPOUT_KEEP_PROB]
                    [--learning_rate_decay LEARNING_RATE_DECAY]
                    [--learning_rate_step LEARNING_RATE_STEP]
                    [--batch_size BATCH_SIZE] [--learning_rate LEARNING_RATE]
                    [--train_steps TRAIN_STEPS] [--save_path SAVE_PATH]
                    [--print_every PRINT_EVERY]
                    [--evaluate_every EVALUATE_EVERY]
                    [--save_every SAVE_EVERY]

optional arguments:
  -h, --help            show this help message and exit
  --test TEST           Path to a .txt file to train on
  --train TRAIN         Path to a .txt file to test on
  --gru_num_hidden GRU_NUM_HIDDEN
                        Number of hidden units in the GRU
  --gru_num_layers GRU_NUM_LAYERS
                        Number of GRU layers in the model
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability
  --learning_rate_decay LEARNING_RATE_DECAY
                        Learning rate decay fraction
  --learning_rate_step LEARNING_RATE_STEP
                        Learning rate step
  --batch_size BATCH_SIZE
                        Number of examples to process in a batch
  --learning_rate LEARNING_RATE
                        Learning rate
  --train_steps TRAIN_STEPS
                        Number of training steps
  --save_path SAVE_PATH
                        Output path for models and dataset files
  --print_every PRINT_EVERY
                        How often to print training progress
  --evaluate_every EVALUATE_EVERY
                        How often the model is evaluated
  --save_every SAVE_EVERY
                        How often to save the model

```

### evaluate model

```

usage: eval.py [-h] --model MODEL --dataset DATASET
               [--gru_num_hidden GRU_NUM_HIDDEN]
               [--gru_num_layers GRU_NUM_LAYERS]
               [--dropout_keep_prob DROPOUT_KEEP_PROB]
               [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Path to model
  --dataset DATASET     Path to the dataset files
  --gru_num_hidden GRU_NUM_HIDDEN
                        Number of hidden units in the GRU
  --gru_num_layers GRU_NUM_LAYERS
                        Number of GRU layers in the model
  --dropout_keep_prob DROPOUT_KEEP_PROB
                        Dropout keep probability
  --batch_size BATCH_SIZE
                        Number of examples to process in a batch
```
