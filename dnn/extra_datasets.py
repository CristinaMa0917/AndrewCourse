from model_package import *
def load_datasets(dataset,isshow):
    # Datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    # make blobs binary
    if dataset == "blobs":
        Y = Y%2
    # 数据可视化
    if isshow:
        plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
        plt.show()
    return X,Y