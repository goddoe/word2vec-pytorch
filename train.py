import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cbow import CBOW
from datasets import WordContextDataset


def train(_=None,
          corpus=None,
          corpus_path=None,
          context_size=2,
          min_word=1,

          embed_dim=100,

          n_epoch=10,
          batch_size=32,
          learning_rate=0.001,
          shuffle=True,
          verbose_iterval=1):

    if _:
        raise Exception("Don't put parameters without keys. Set parameters with the key together.")

    # Load data
    wcd = WordContextDataset(corpus=corpus,
                             corpus_path=corpus_path,
                             context_size=context_size,
                             min_word=min_word)

    data_loader = DataLoader(wcd,
                             batch_size=batch_size,
                             shuffle=shuffle)

    # Model
    cbow = CBOW(vocab_size=wcd.vocab_size,
                embed_dim=embed_dim)

    # Training Parameters
    optimizer = optim.SGD(cbow.parameters(),
                          lr=learning_rate)
    loss_fn = nn.NLLLoss()
    loss_list = []

    # Use GPU, if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cbow.to(device)

    for epoch_i in range(n_epoch):
        for batch_i, (X, Y) in enumerate(data_loader):
            X, Y = X.to(device), Y.to(device)
            cbow.zero_grad()

            pred_log_prob = cbow(X)

            loss = loss_fn(pred_log_prob, Y)

            loss.backward()
            loss_list.append(float(loss.to('cpu').data.numpy()))

            optimizer.step()

            if epoch_i % verbose_iterval == 0:
                print("loss : {:.3f}".format(loss_list[-1]))

    return {'wcd': wcd,
            'cbow': cbow,
            'loss_list': loss_list,
            'data_loader': data_loader}


