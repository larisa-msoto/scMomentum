import scmomentum as scm

global dset, clustcol
dset = "hFB18"
clustcol = "labels"

adata_dir = "/Users/larisamorales/Documents/KAUST/scgrn-project/objects/"
adata_path = adata_dir + "scvelo/" + dset + "-adata.pkl"
adata = scm.utilities.load_adata(adata_path)

adata = scm.network_inference.preprocess(adata, clustcol)
clusters = set(adata.obs[clustcol])

for c in clusters:

    print(c)
    scm.network_inference.predict_network(
        adata, cluster=c, genes="highexp", network_size=100, clustcol=clustcol
    )
    print(adata.uns["W-highexp-100"][c].shape)
