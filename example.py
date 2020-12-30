from scmomentum.network_inference import *

global dset, clustcol
dset = "hFB18"
clustcol = "labels"

adata_dir = "/Users/larisamorales/Documents/KAUST/scgrn-project/objects/"
adata_path = adata_dir + "scvelo/" + dset + "-adata.pkl"
adata = load_adata(adata_path)

adata = preprocess(adata, clustcol)
clusters = set(adata.obs[clustcol])

for c in clusters:

    print(c)
    predict_network(
        adata, cluster=c, genes="highexp", network_size=100, clustcol=clustcol
    )
    print(adata.uns["W-highexp-100"][c].shape)
