import regex as re
import torch, time, argparse, os
import torch.utils.data as td
import torch.nn as nn
from collections import defaultdict, Counter

def main():
  # Read data
  data_train = read_files('data/x_train.txt', 'data/y_train.txt')
  data_test = read_files('data/x_test.txt', 'data/y_test.txt')
  features = select_ngrams(data_train, n=config.N, amount=20)
  train_set, test_set, meta = construct_datasets(data_train, data_test, features)

  # Init model
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = MLP(meta['input_size'], meta['output_size']).to(device)
  loss_fn = nn.CrossEntropyLoss()
  opt = torch.optim.Adam(model.parameters())

  for epoch in range(config.epochs):
    start_time = time.time()
    total_loss = 0
    total_accuracy = 0
    for batch, (x, t) in enumerate(train_set):
      x = x.to(device)
      t = t.to(device)

      # Forward
      y = model(x)
      loss = loss_fn(y, t)

      # Backward
      opt.zero_grad()
      loss.backward()
      opt.step()

      # Information
      total_loss += loss.item()
      total_accuracy += (t == y.argmax(1)).sum().item() / len(t)
      rate = (batch + 1) / (time.time() - start_time)
      print('\rEpoch {:03d}/{:03d}, Batch {:05d}/{:05d} ({:.2f}/s) | Loss: {:.4f}, Accuracy: {:.1f}%'.format(epoch+1, config.epochs, batch+1, len(train_set), rate, total_loss/(batch+1), (total_accuracy/(batch+1))*100), end='')

    # Evaluate on test set
    test_accuracy, class_accuracy = evaluate(test_set, meta, model, device)

    print(' | Test accuracy: {:.2f}%'.format(test_accuracy*100))

    # Save checkpoint
    state = {
      'epoch': epoch + 1,
      'model': model.state_dict(),
      'opt': opt.state_dict(),
      'train_accuracy': total_accuracy,
      'train_loss': total_loss,
      'test_accuracy': test_accuracy,
      'classes': class_accuracy
    }

    torch.save(state, 'models/{}-{}'.format(config.name, epoch+1))

  # %% Create text
  write_results('{}-{}'.format(config.name, epoch+1))

# Model
class MLP(nn.Module):
  def __init__(self, n_inputs, n_targets):
    super().__init__()

    self.layers = nn.ModuleList([
      nn.Linear(n_inputs, 512),
      nn.ReLU(),
      nn.Linear(512, n_targets)
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

# Helper functions
def write_results(checkpoint):
  classes = torch.load('models/'+checkpoint)['classes']
  text = 'language\tprecision\trecall\tF1\n'
  for language, data in classes.items():
    try:
      precision = data['right']/data['times_predicted']
      recall = data['right']/(data['right']+data['wrong'])
      f1 = (2*precision*recall)/(precision+recall)
    except:
      precision = recall = f1 = 0
    text += '{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(language, precision, recall, f1)

  with open('results/{}-class-results.txt'.format(checkpoint), 'w+') as f:
    f.write(text)

def read_files(path_x, path_y, filter=None):
  # input:
  #  path_x: (str) path of x (paragraphs)
  #  path_y: (str) path of y (languages)
  # return:
  #   dict: {language: [paragraph, ... ,paragraph], language: [...}
  with open(path_x, 'r') as f:
    paragraphs = f.read().strip().split('\n')

  with open(path_y, 'r') as f:
    languages = f.read().strip().split('\n')

  data = defaultdict(lambda: [])
  for lang, para in zip(languages, paragraphs):
    if filter is None or lang in filter:
      para = re.sub(r'\p{P}', '', para) # Remove punctuation
      data[lang].append(para)
  return data

def select_ngrams(data, n=3, amount=20):
  # input:
  #  data: (dict) paragraphs grouped per language
  #  n: (int) size of n-gram
  #  amount: (int) number of ngrams to use per language
  # return:
  #   dict: set(ngram, ngream, ..., ngram)
  ngrams = []
  for language, paragraphs in data.items():
    paragraph_grams = [set(para[i:i+n] for i in range(len(para)-n)) for para in paragraphs]
    c = Counter([item for sublist in paragraph_grams for item in list(sublist)])
    ngrams += [x[0] for x in c.most_common(amount)]

  return set(ngrams)

def construct_datasets(data_train, data_test, features):
  meta = {
    'i2x': {index: feature for index, feature in enumerate(features)},
    'x2i': {feature: index for index, feature in enumerate(features)},
    'i2t': {index: lang for index, lang in enumerate(data_train.keys())},
    't2i': {lang: index for index, lang in enumerate(data_train.keys())},
    'input_size': len(features),
    'output_size': len(data_train.keys())
  }

  dataloaders = []
  for data in (data_train, data_test):
    num_paragraphs = sum([len(data[x]) for x in data.keys()])

    X = torch.FloatTensor(num_paragraphs, meta['input_size'])
    t = torch.LongTensor(num_paragraphs)

    i = 0
    for j, (language, paragraphs) in enumerate(data.items()):
      language_index = meta['t2i'][language]
      for para in paragraphs:
        X[i, :] = extract_features(para, meta['x2i'])
        t[i] = language_index
        i += 1
      print('\rExtracting features, language {}/{}'.format(len(dataloaders)*len(data)+j+1, len(data)*2), end='')

    dataloaders.append(td.DataLoader(td.TensorDataset(X, t), shuffle=True, batch_size=config.batch_size))

  return dataloaders[0], dataloaders[1], meta

def extract_features(paragraph, x2i):
  out = torch.zeros(1, len(x2i))
  ngrams = [paragraph[i:i+config.N] for i in range(len(paragraph)-config.N)]
  num_ngrams = len(ngrams)
  c = Counter(ngrams)

  for feature, index in x2i.items():
    if feature in c:
      out[0, index] = c[feature] / num_ngrams

  return out

def evaluate(dataset, meta, model, device):
  class_accuracy = defaultdict(lambda: {'right': 0, 'wrong': 0, 'predictions': [], 'times_predicted': 0})
  for x, t in dataset:
    x = x.to(device)
    t = t.to(device)

    y = model(x)

    for yi, ti in zip(y.argmax(1), t):
      class_accuracy[meta['i2t'][yi.item()]]['times_predicted'] += 1
      if ti == yi:
        class_accuracy[meta['i2t'][ti.item()]]['right'] += 1
      else:
        class_accuracy[meta['i2t'][ti.item()]]['predictions'].append(meta['i2t'][yi.item()])
        class_accuracy[meta['i2t'][ti.item()]]['wrong'] += 1

  class_accuracy['everything']
  for lang, value in class_accuracy.items():
    if lang != 'everything':
      value['accuracy'] = value['right'] / (value['right'] + value['wrong'])
      value['predictions'] = Counter(value['predictions'])
      class_accuracy['everything']['right'] += value['right']
      class_accuracy['everything']['wrong'] += value['wrong']
      class_accuracy['everything']['times_predicted'] += value['times_predicted']
  class_accuracy['everything']['accuracy'] = class_accuracy['everything']['right'] / (class_accuracy['everything']['right'] + class_accuracy['everything']['wrong'])

  return class_accuracy['everything']['accuracy'], dict(class_accuracy)

if __name__ == '__main__':
  # Create directory structure
  if not os.path.exists('results'):
    os.mkdir('results')

  if not os.path.exists('models'):
    os.mkdir('models')

  # Parse training configuration
  parser = argparse.ArgumentParser()

  # Model params
  parser.add_argument('--name', type=str, required=True, help='Name of model')
  parser.add_argument('--N', type=int, default=3, help='Order of n-gram')
  parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')
  parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')

  config = parser.parse_args()

  main()
