import argparse
import pickle

def read(config):
    dic = {}
    with open(config.input) as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            dic[line[0]] = line[3]
    pickle.dump(dic, open(config.output,'wb'))

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input', type=str, required=True, help="file that contains scores from sklearn summary")
    parser.add_argument('--output', type=str, required=True, help="filename of dict")
    config = parser.parse_args()

    # Train the model
    read(config)
