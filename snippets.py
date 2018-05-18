import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cbow import CBOW
from datasets import WordContextDataset

# tiny_corpus from pytorch tutorial (https://pytorch.org/tutorials/)
tiny_corpus = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold."""

# Load data
wcd = WordContextDataset(corpus=tiny_corpus,
                         context_size=2,
                         min_word=1)

data_loader = DataLoader(wcd, batch_size=16, shuffle=True)

# Model
cbow = CBOW(vocab_size=wcd.vocab_size,
            embed_dim=100)

# Training Parameters
n_epoch = 1000
learning_rate = 0.001

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
        loss_list.append(loss.to('cpu').data.numpy())

        optimizer.step()


# ======================================
# Snippets for visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Evaluation embeddings
word_embeddings = cbow.embeddings.to('cpu').weight.data.numpy()

fig_loss = plt.figure("loss vs batch")
ax_loss = fig_loss.add_subplot(111)
ax_loss.plot(loss_list, color='blue')
ax_loss.set_xlabel('mini-batch iter')
ax_loss.set_ylabel('loss')


# PCA or TSNE Visualization
# dim_reduction = PCA(n_components=2)
dim_reduction = TSNE(n_components=2, init='pca')

compressed = dim_reduction.fit_transform(word_embeddings)

fig = plt.figure("Visualize Embeddings")
ax = fig.add_subplot(111)
ax.scatter(compressed[:, 0], compressed[:, 1])

idx_word_tuple_list = sorted(wcd.idx_word_dict.items(),
                             key=lambda t: t[1])

for i, (idx, word) in enumerate(idx_word_tuple_list):
    ax.annotate(word, compressed[i])
