import numpy as np
import pandas as pd
import scvelo as scv
from collections import defaultdict
from scmomentum.distances import expression_distance,network_distance

def preprocess(adata, clustcol):

	"""
	first step
	"""

	# Compute cluster distances

	expression_distance(adata, clustcol)

	# Remove genes with NaN's in velocity

	V = adata.layers["velocity"]
	genes_valid = (
		adata.var[["velocity_genes"]]
		.iloc[np.where(np.logical_not(np.isnan(V.sum(axis=0))))[0].tolist(), :]
		.index
	)
	adata_valid = adata[:, genes_valid]
	adata_valid.var["gtype"] = "Velocity_not_nan"

	return adata


def rank_genes(V, X, g, n):

	# INPUT
	# V = velocity matrix for the cells in a specific cluster
	# X = expression matrix for the cells in a specific cluster
	# n = number of genes to choose
	# g = str ranking method. Options:
	#   * absvel = gene ranking based on the mean (across cells) absolute value of velocity
	#   * topvel = gene ranking based on the mean (across cells) value of velocity (including sign)
	#   * stdvel = gene ranking based on decreasing standard deviation of velocity across cells
	#   * random = gene set selected at random
	#   * stdexp = gene ranking basedon decreasing standard deviation of expression across cells
	#   * highexp = gene ranking basedon decreasing expression level across cells
	# OUTPUT
	# genes = list with genes selected (based on the cluster matrix dimensions)
	
	options = ['absvel','topvel','stdvel','stdexp','highexp']

	if g not in options:
		raise ValueError("Gene mode is not in options. Please indicate a valid gene mode: absvel,topvel,stdvel,stdexp,highexp")
	

	if g == "absvel":
		gset = V.dropna().abs().mean(0).dropna().sort_values(ascending=False)[0:n].index.tolist()
	elif g == "topvel":
		gset = V.dropna(0).dropna().sort_values(ascending=False)[0:n].index.tolist()
	elif g == "stdvel":
		gset = V.dropna().std(0).dropna().sort_values(ascending=False).index.tolist()[0:n]
	elif g == "stdexp":
		gset = X.dropna().std(0).dropna().sort_values(ascending=False).index.tolist()[0:n]
	elif g == "highexp":
		gset = X.dropna().mean(0).dropna().sort_values(ascending=False).index.tolist()[0:n]

	return gset


def predict_network(adata, cluster, genes, network_size, clustcol,name='',layer='spliced',copy=False):

	## INPUT

	#  clustcol - cluster column used
	# cluster - individual cluster label
	# genes - either a str (vrank,maxstd,topvar,abstop,random) or a list of genes (same as in adata.var.index)
	# network_size - number of genes used to infer the network
	# copy - whether a copy of the network should be returned. By default copy=True

	# Cluster cells

	ind = adata.obs[clustcol] == cluster
	V = pd.DataFrame(adata.layers["velocity"][ind.values, :], columns=adata.var.index)
	X = pd.DataFrame(
		adata.layers[layer][ind.values, :].todense(), columns=adata.var.index
	)

	# Get  genes

	if isinstance(genes, str):

		if (genes!='allvalid'):
			geneset = rank_genes(V, X, genes, network_size)
			tag = genes + "-" + str(network_size) + name
		else:
			geneset = adata.var['velocity_genes'].index[adata.var['velocity_genes']].sort_values().tolist()
			tag = genes + "-" + str(network_size) + name

	else:
		geneset = [g for g in genes if g in X.columns]
		genes = "manual"
		tag = genes + "-" + name

	# Infer networks

	Xc = X.loc[:, geneset]
	Xpinv = np.linalg.pinv(Xc)
	Vc = V.loc[:, geneset]
	Gf = np.diag(adata.var.fit_gamma.loc[geneset,])
	W = np.nan_to_num(np.dot(Xpinv, (Vc + np.dot(Xc, Gf))), nan=0)
	W = pd.DataFrame(np.array(W, dtype=np.float64), index=geneset, columns=geneset)

	if copy == False:

		if tag in adata.uns.keys():
			adata.uns[tag][cluster] = W
			print('--> Updated adata.uns with key ','\''+tag+':'+cluster+'\'')
		else:
			adata.uns[tag] = defaultdict(pd.DataFrame)
			adata.uns[tag][cluster] = W
			print('--> Updated adata.uns with key ','\''+tag+':'+cluster+'\'')

	else:
		return W
