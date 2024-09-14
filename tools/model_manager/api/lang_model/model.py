import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .utils import AvgrageMeter, accuracy, adjust_learning_rate

class RelationModel(nn.Module):
    def __init__(self, hidden_dim, embed_dim, vocab_size, 
                 device, num_layers=2, dropout=0.5):
        super(RelationModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, 
                            dropout=dropout, num_layers=num_layers)
        self.decoder = nn.Linear(2*hidden_dim, vocab_size)
        self.device=device
        
    def forward(self, inputs, lengths, hidden=None):
        self.lstm.flatten_parameters()
        batch_size, seq_len  = inputs.size()
        embeds = self.embedding(inputs)
        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        
        if not hidden:
            h0 = torch.zeros(2*self.num_layers, batch_size, self.hidden_dim, device=self.device).to(self.device)
            c0 = torch.zeros(2*self.num_layers, batch_size, self.hidden_dim, device=self.device).to(self.device)
            hidden = (h0, c0)
            
        states, hidden = self.lstm(embeds, hidden)
        states, _ = pad_packed_sequence(states, total_length=seq_len, batch_first=True)

        outputs = self.decoder(states).reshape(batch_size*seq_len, -1)
        return outputs


def train(train_loader, test_loader, model, 
                model_path, num_epoch, optimizer, 
                lr, criterion, device):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    top10 = AvgrageMeter()

    best_topN = -1
    best_model = None
    shape = None
    
    for epoch in range(num_epoch):
        model.train()
        adjust_learning_rate(optimizer, epoch, lr)
        train_loss = 0.0
        for i, (inputs, labels, lengths) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            lengths, sorted_indices = torch.sort(lengths, descending=True)
            sorted_indices = sorted_indices.to(device)
            inputs = inputs.index_select(0, sorted_indices).to(device)
            labels = labels.index_select(0, sorted_indices).reshape(-1,).to(device)
            
            outputs = model(inputs, lengths)
            
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prec1, prec5, prec10 = accuracy(outputs, labels, lengths, (1,5,10), device)
            n = inputs.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            top10.update(prec10.item(), n)
            train_loss += loss.item()

        if epoch % 10 == 0:
            print(f"\nepoch {epoch} train, train loss: {train_loss/(i+1)}, top1_acc: {top1.avg}, top5_acc: {top5.avg}, top10_acc: {top10.avg}")
            eval_top1, eval_top5, eval_top10 = eval(test_loader, model, device)
            print(f"\nepoch {epoch} eval, top1_acc: {eval_top1}, top5_acc: {eval_top5}, top10_acc: {eval_top10}")
            traced_script_module = torch.jit.trace(model, (inputs, lengths))
            traced_script_module.save(f"{model_path}/syscall_model_jit_epoch_{epoch}.pt")
            torch.save(model.state_dict(), f'{model_path}/syscall_model_epoch_{epoch}.model')
            if eval_top10 > best_topN:
                best_topN = eval_top10
                best_model = copy.deepcopy(model)
                shape = (inputs, lengths)

    traced_script_module = torch.jit.trace(best_model, shape)
    traced_script_module.save(f"{model_path}/syscall_model_jit_best.pt")
    torch.save(model.state_dict(), f'{model_path}/syscall_model_best.model')
    
    return f"{model_path}/syscall_model_jit_best.pt" if best_model else None


def eval(data_loader, model, device):
    model.eval()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    top10 = AvgrageMeter()
    with torch.no_grad():
        for i, (inputs, labels, lengths) in enumerate(data_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            lengths, sorted_indices = torch.sort(lengths, descending=True)
            sorted_indices = sorted_indices.to(device)
            inputs = inputs.index_select(0, sorted_indices).to(device)
            labels = labels.index_select(0, sorted_indices).reshape(-1,).to(device)

            outputs = model(inputs, lengths).to(device)
            prec1, prec5, prec10 = accuracy(outputs, labels, lengths, (1,5,10), device)

            n = inputs.size(0)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            top10.update(prec10.item(), n)

    return top1.avg, top5.avg, top10.avg
