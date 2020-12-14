import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gzip import GzipFile
from collections import defaultdict, Counter
from tqdm import tqdm
#from word2gauss import GaussianEmbedding, iter_pairs
#from words import Vocabulary

def tokenizer(s):
    '''
    Whitespace tokenizer
    '''
    return s.lower().replace(".", "").replace(",", "").replace(":", "").replace(";", "").strip().split()



#### NEW ####
with open('christmas_carol.txt', 'r') as file:
    data = tokenizer(file.read().replace('\n', ' '))

#print(data)

entity_2_idx = defaultdict(lambda: len(entity_2_idx))
counter = Counter()
dataset = []

for entity in tqdm(data):
    entity_2_idx[entity]
    counter[entity_2_idx[entity]] += 1
    dataset.append(entity_2_idx[entity])

print(entity_2_idx)
print("\n\n")
print(counter)
print("\n\n")
print(dataset)

#### OLD ####

# load the vocabulary
vocab = Vocabulary(entity_2_idx)

# create the embedding to train
# use 100 dimensional spherical Gaussian with KL-divergence as energy function
embed = GaussianEmbedding(len(vocab), 100,
    covariance_type='spherical', energy_type='KL')

# open the corpus and train with 8 threads
# the corpus is just an iterator of documents, here a new line separated
# gzip file for example
with GzipFile('location_of_corpus', 'r') as corpus:
    embed.train(iter_pairs(corpus, vocab), n_workers=8)

# save the model for later
embed.save('model_file_location', vocab=vocab.id2word, full=True)
