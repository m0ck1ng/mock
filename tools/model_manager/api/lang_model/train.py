import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import read_corpus, read_syscalls
from dataset import Vocab, SysDataset
from model import RelationModel, train, eval

def test(argv):
    corpus_file = argv[1]
    testcase_file = argv[2]
    model_path = argv[3]

    syscalls = read_syscalls(r"/home/data/syscall_names")
    corpus = read_corpus(corpus_file)

    vocab = Vocab()
    vocab.addDict(syscalls)
    vocab.addCorpus(corpus)
    sys_dataset = SysDataset(corpus, vocab)

    train_size =  int(0.8*len(sys_dataset))
    test_size = len(sys_dataset)-train_size
    train_dataset, test_dataset = torch.utils.data.random_split(sys_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lr = 0.001
    embed_dim = 64
    hidden_dim = 128

    model = RelationModel(hidden_dim, embed_dim, len(sys_dataset.vocab), device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    num_epoch = 50
    model_file = train(train_loader, test_loader, model, \
        model_path, num_epoch, optimizer, \
            lr, criterion, device)

if __name__ == "__main__":
    test(sys.argv)
