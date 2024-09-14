import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
# import h5py

class Vocab:
    def __init__(self):
        self.idx2word = ["PAD","SOS","EOS"]
        self.word2idx = {"PAD":0,"SOS":1,"EOS":2}
        self.word2count = {"PAD":1,"SOS":1,"EOS":1}
        self.num_words = 3
        
    def addCorpus(self, corpus):
        for sent in corpus:
            for word in sent:
                self.addWord(word)

    def addDict(self, dictionary):
        for word in dictionary:
            if word not in self.word2idx:
                self.word2idx[word] = self.num_words
                self.word2count[word] = 0
                self.idx2word.append(word)
                self.num_words += 1

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.word2count[word] = 1
            self.idx2word.append(word)
            self.num_words += 1
        else:
            self.word2count[word] += 1
    
    def __len__(self):
        return self.num_words


class SysDataset(Dataset):
    def __init__(self, corpus=None, vocab=None, max_len=20):
        self.vocab = vocab
        self.max_len = max_len
        self.seq_len = []
        self.corpus = []
        if corpus and vocab:
            self.corpus = np.array([self._sent2data(sent) for sent in corpus], dtype=np.int)
            self.seq_len = np.array(self.seq_len, dtype=np.int)

    
    def _sent2data(self, sent):
        self.seq_len.append(min(len(sent), self.max_len))
        if len(sent) > self.max_len:
            sent = ["SOS"]+sent[:self.max_len]+["EOS"]
        else:
            sent = ["SOS"]+sent+['PAD' for _ in range(self.max_len-len(sent))]+["EOS"]
        data = [self.vocab.word2idx[word] for word in sent]
        return data
    
    def __getitem__(self, idx):
        return (torch.tensor(self.corpus[idx][:-1], dtype=torch.long), 
                torch.tensor(self.corpus[idx][1:], dtype=torch.long), 
                self.seq_len[idx]+1)
    
    def __len__(self):
        return len(self.corpus)

    def save(self, corpus_file, vocab_file):
        with h5py.File(corpus_file, 'w') as f:
            f['corpus'] = self.corpus
            f['seq_len'] = self.seq_len
        
        with open(vocab_file, 'wb') as f:
            pickle.dump(obj=self.vocab.word2idx, file=f)
    
    def load(self, corpus_file, vocab_file):
        with h5py.File(corpus_file, 'r') as f:
            self.corpus = f['corpus'].value
            self.seq_len = f['seq_len'].value
            self.max_len = self.corpus.shape[1]
        
        self.vocab = Vocab()
        with open(vocab_file, 'rb') as f:
            self.vocab.word2idx = pickle.load(file=f)
            items = sorted(self.vocab.word2idx.items(), key=lambda x: x[1])
            self.vocab.idx2word = [item[0] for item in items]
            self.vocab.num_words = len(self.vocab.idx2word)