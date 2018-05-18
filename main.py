import argparse
from train import train

KEY_TSNE = "tsne"
KEY_PCA = "pca"


def visualize(result_dict, dim_reduction_type=KEY_TSNE):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    r = result_dict
    wcd, cbow, loss_list = r['wcd'], r['cbow'], r['loss_list']

    # Evaluation embeddings
    word_embeddings = cbow.embeddings.to('cpu').weight.data.numpy()

    fig_loss = plt.figure("loss vs batch")
    ax_loss = fig_loss.add_subplot(111)
    ax_loss.plot(loss_list, color='blue')
    ax_loss.set_xlabel('mini-batch iter')
    ax_loss.set_ylabel('loss')

    # PCA or TSNE Visualization
    if KEY_TSNE:
        dim_reduction = TSNE(n_components=2, init='pca')
    elif KEY_PCA:
        dim_reduction = PCA(n_components=2)
    else:
        raise Exception("Wrong dim_reduction_type")

    compressed = dim_reduction.fit_transform(word_embeddings)

    fig = plt.figure("Visualize Embeddings")
    ax = fig.add_subplot(111)
    ax.scatter(compressed[:, 0], compressed[:, 1])

    idx_word_tuple_list = sorted(wcd.idx_word_dict.items(),
                                 key=lambda t: t[1])

    for i, (idx, word) in enumerate(idx_word_tuple_list):
        ax.annotate(word, compressed[i])


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='preprocess data')
    parser.add_argument('--corpus_path',
                        type=str,
                        help='corpus path')
    parser.add_argument('--context_size',
                        type=int,
                        default=2,
                        help='context size')

    parser.add_argument('--min_word',
                        type=int,
                        default=1,
                        help='minimum word')

    parser.add_argument('--embed_dim',
                        type=int,
                        default=100,
                        help='embedding dimension')

    parser.add_argument('--n_epoch',
                        type=int,
                        default=10,
                        help='context_size')

    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='context_size')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='learning rate')

    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help='flag shuffle mini batches')

    parser.add_argument('--verbose_iterval',
                        type=int,
                        default=5,
                        help='Interval for print loss.')

    parser.add_argument('--vis',
                        type=bool,
                        default=False,
                        help='Interval for print loss.')

    parser.add_argument('--dim_reduction_type',
                        type=str,
                        default=KEY_TSNE,
                        help='Interval for print loss.')

    p = parser.parse_args()

    print("train start")
    result_dict = train(corpus_path=p.corpus_path,
                        context_size=p.context_size,
                        min_word=p.min_word,
                        embed_dim=p.embed_dim,
                        n_epoch=p.n_epoch,
                        batch_size=p.batch_size,
                        learning_rate=p.learning_rate,
                        shuffle=p.shuffle,
                        verbose_iterval=p.verbose_iterval)
    print("train end")

    visualize(result_dict, p.dim_reduction_type)

