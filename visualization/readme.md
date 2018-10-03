### extract accuracies from sklearn model summary txt

```
usage: extrac_acc.py [-h] --input INPUT --output OUTPUT

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    file that contains scores from sklearn summary
  --output OUTPUT  filename of dict
```


### visualize accuracies in countbased t-sne plot

```
usage: visualization.py [-h] --train TRAIN --legal_chars LEGAL_CHARS
                        --acc_dict ACC_DICT --save_fig SAVE_FIG
                        [--label_threshold LABEL_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Path to the preprocessed test data
  --legal_chars LEGAL_CHARS
                        Path legal chars
  --acc_dict ACC_DICT   Path to accuracy dict
  --save_fig SAVE_FIG   Path for saving figure
  --label_threshold LABEL_THRESHOLD
                        acc threshold for showing labels
```