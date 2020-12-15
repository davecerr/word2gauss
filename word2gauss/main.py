import numpy as np
import os
import pickle as pkl
import time
import gzip
import json

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gzip import GzipFile
from collections import defaultdict, Counter
from tqdm import tqdm
from embeddings import GaussianEmbedding #, iter_pairs
from words import Vocabulary, iter_pairs

def _open_file(filename):
    with gzip.open(filename) as infile:
        for _, line in enumerate(infile):
            yield json.loads(line)

def tokenizer(s):
    '''
    Whitespace tokenizer
    '''
    return s.lower().replace(".", "").replace(",", "").replace(":", "").replace(";", "").strip().split()



#### NEW ####
# filename = 'war_and_peace.txt'
#
# with open(filename, 'r') as file:
#     data = tokenizer(file.read().replace('\n', ' '))

################################################################################


print("\n\n----------- LOADING DATA ----------")
if os.path.exists("data.pkl"):
    start = time.time()
    print("loading from existing pickle")
    pickle_in = open("data.pkl","rb")
    data = pkl.load(pickle_in)
    end = time.time()
    print("loaded in {} secs".format(round(end - start,2)))
else:
    print("loading from gzip files")
    files = []
    for _, _, fs in os.walk("data/", topdown=False):
        files += [f for f in fs if f.endswith(".gz")]

    files = [os.path.join("data/page_dist_training_data/", f) for f in files]
    data_list = []
    for i, file in tqdm(enumerate(files)):
        sentences = list(_open_file(file))
        data_list += sentences

    data = ""
    for lst in tqdm(data_list):
        for entity in lst:
            data += entity


    pickle_out = open("data.pkl","wb")
    pkl.dump(data, pickle_out)
    pickle_out.close()

# print("Corpus length = {}".format(len(corpus)))


################################################################################








# print(data)

entity_2_idx = defaultdict(lambda: len(entity_2_idx))
counter = Counter()
dataset = []

for entity in tqdm(data):
    entity_2_idx[entity]
    counter[entity_2_idx[entity]] += 1
    dataset.append(entity_2_idx[entity])

# print(entity_2_idx)
num_tokens = len(entity_2_idx)
print("num_tokens = {}".format(num_tokens))


# print(entity_2_idx)
# print("\n\n")
# print(counter)
# print("\n\n")
# print(dataset)

#### OLD ####

# load the vocabulary
vocab = Vocabulary(entity_2_idx,tokenizer)
# print(vocab)
# create the embedding to train
# use 100 dimensional spherical Gaussian with KL-divergence as energy function
embed = GaussianEmbedding(num_tokens, 100,
    covariance_type='diagonal', energy_type='KL')


print("---------- INITIAL EMBEDDING MEANS ----------")
print(embed.mu)
print("---------- INITIAL EMBEDDING COVS ----------")
print(embed.sigma)

# open the corpus and train with 8 threads
# the corpus is just an iterator of documents, here a new line separated
# gzip file for example

"""

with open(filename, 'r') as corpus:
    # for pair in iter_pairs(corpus, vocab):
        # print(pair.shape)
    embed.train(iter_pairs(corpus, vocab), n_workers=8)


print("---------- FINAL EMBEDDING MEANS ----------")
print(embed.mu)
print("---------- FINAL EMBEDDING COVS ----------")
print(embed.sigma)

sigma_norms = np.linalg.norm(embed.sigma, axis=1)
most_general_indices = np.split(sigma_norms,2)[0].argsort()[-10:][::-1]
most_specific_indices = np.split(sigma_norms,2)[0].argsort()[:10]

idx_2_entity = {v: k for k, v in entity_2_idx.items()}

print("MOST GENERAL ENTITIES")
for idx in most_general_indices:
    print(idx_2_entity[idx])

print("MOST SPECIFIC ENTITIES")
for idx in most_specific_indices:
    print(idx_2_entity[idx])
"""
# save the model for later
# embed.save('model_file_location', vocab=vocab.id2word, full=True)
