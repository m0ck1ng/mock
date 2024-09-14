import os
import sys
import torch
import torch.nn as nn
import random
import datetime
from torch.utils.data import DataLoader
import torch.optim as optim

from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt

from .lang_model.utils import read_corpus, read_syscalls
from .lang_model.dataset import Vocab, SysDataset
from .lang_model.model import RelationModel, train

@csrf_exempt
def model_train(request):
    if request.method == "POST":
        now_time = datetime.datetime.now()
        corpus_dir = request.POST.get("corpus", None)
        testcase_dir = request.POST.get("testcase", None)

        if not corpus_dir or not os.path.isdir(corpus_dir):
            return HttpResponse("error: invalid parameters")

        syscalls = read_syscalls(f"{settings.BASE_DIR}/api/lang_model/data/syscall_names")
        corpus = read_corpus(corpus_dir)
        
        # sample_size = int(0.6*len(corpus))
        # testcase_sample = read_corpus(testcase_dir, sample_size)
        # corpus.extend(testcase_sample)

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

        num_epoch = 50
        lr = 0.001
        embed_dim = 64
        hidden_dim = 128

        model = RelationModel(hidden_dim, embed_dim, len(vocab), device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model_path = f"{settings.BASE_DIR}/api/lang_model/checkpoints"
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_file = train(train_loader, test_loader, model, \
            model_path, num_epoch, optimizer, \
            lr, criterion, device)

        time_cost = datetime.datetime.now() - now_time
        print(f"model_update_time_cost: {time_cost}")
        
        return HttpResponse(model_file)
        
