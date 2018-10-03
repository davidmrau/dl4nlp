# %% Imports
import regex as re
import torch, time
import torch.utils.data as td
import torch.nn as nn
from collections import defaultdict, Counter

# %% Parameters
NAME = 'testgram-20'
EPOCHS = 20
BATCH_SIZE = 32
N = 3

torch.manual_seed(0)
torch.cuda.manual_seed(0)

# %% Data loading functions
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

def construct_dataset(data, features, x2i=None, t2i=None):
  if not t2i:
    t2i = {lang: index for index, lang in enumerate(data.keys())}
  i2t = {index: lang for lang, index in t2i.items()}

  if not x2i:
    x2i = {feature: index for index, feature in enumerate(features)}
  i2x = {index: feature for feature, index in x2i.items()}

  num_features = len(x2i)
  num_languages = len(t2i)
  num_paragraphs = sum([len(data[x]) for x in data.keys()])

  X = torch.FloatTensor(num_paragraphs, num_features)
  t = torch.LongTensor(num_paragraphs)

  i = 0
  for language, paragraphs in data.items():
    language_index = t2i[language]
    for para in paragraphs:
      X[i, :] = extract_features(para, x2i)
      t[i] = language_index
      i += 1

  out = td.DataLoader(td.TensorDataset(X, t), shuffle=True, batch_size=BATCH_SIZE)
  meta = {'i2x': i2x, 'i2t': i2t, 'x2i': x2i, 't2i': t2i, 'num_features': num_features, 'num_languages': num_languages}

  return out, meta

def extract_features(paragraph, x2i):
  out = torch.zeros(1, len(x2i))
  ngrams = [paragraph[i:i+N] for i in range(len(paragraph)-N)]
  num_ngrams = len(ngrams)
  c = Counter(ngrams)

  for feature, index in x2i.items():
    if feature in c:
      out[0, index] = c[feature] / num_ngrams

  return out

# %% Create dataset
filter = None #set(['wuu', 'zh-yue', 'zho'])
data_train = read_files('data/x_train.txt', 'data/y_train.txt', filter)
data_test = read_files('data/x_test.txt', 'data/y_test.txt', filter)
features = select_ngrams(data_train, n=N, amount=20)

training_set, meta = construct_dataset(data_train, features)
test_set, _ = construct_dataset(data_test, features, x2i=meta['x2i'], t2i=meta['t2i'])

# torch.save(meta, 'ngram_meta.pt')
# torch.save(training_set, 'ngram_train.pt')
# torch.save(test_set, 'ngram_test.pt')

# %% Load dataset
# meta = torch.load('ngram_meta.pt')
# training_set = torch.load('ngram_train.pt')
# test_set = torch.load('ngram_test.pt')

# %% Model
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

def evaluate(dataset, meta):
  class_accuracy = defaultdict(lambda: {'right': 0, 'wrong': 0, 'predictions': [], 'times_predicted': 0})
  for x, t in test_set:
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

# %% Training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MLP(meta['num_features'], meta['num_languages']).to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

for epoch in range(EPOCHS):
  start_time = time.time()
  total_loss = 0
  total_accuracy = 0
  for batch, (x, t) in enumerate(training_set):
    x = x.to(device)
    t = t.to(device)

    y = model(x)
    loss = loss_fn(y, t)
    opt.zero_grad()
    loss.backward()
    opt.step()

    total_loss += loss.item()
    total_accuracy += (t == y.argmax(1)).sum().item() / len(t)
    rate = (batch + 1) / (time.time() - start_time)
    print('\rEpoch {:03d}/{:03d}, Batch {:05d}/{:05d} ({:.2f}/s) | Loss: {:.4f}, Accuracy: {:.1f}%'.format(epoch+1, EPOCHS, batch+1, len(training_set), rate, total_loss/(batch+1), (total_accuracy/(batch+1))*100), end='')

  test_accuracy, class_accuracy = evaluate(test_set, meta)

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

  torch.save(state, '{}-{}'.format(NAME, epoch+1))

# %% Create text
checkpoint = torch.load('{}-{}'.format(NAME, EPOCHS))['classes']
text = 'language\tprecision\trecall\tF1\n'
for language, data in checkpoint.items():
  precision = data['right']/data['times_predicted']
  recall = data['right']/(data['right']+data['wrong'])
  f1 = (2*precision*recall)/(precision+recall)
  text += '{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(language, precision, recall, f1)

with open('{}-result.txt'.format(NAME), 'w+') as f:
  f.write(text)
  print(text)
