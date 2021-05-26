import scipy as scp
import matplotlib.cm as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.decomposition as skd


def padMatrix(n,m):
    A=scp.sparse.csr_matrix((n*m,n*m),dtype=float)
    for i in range(n*m):

        if i==0:
            A[i,i+1]=1/3
            A[i,i+m]=1/3
            A[i,i+m+1]=1/3
        elif i==m-1:
            A[i,i-1]=1/3
            A[i,i+m-1]=1/3
            A[i,i+m]=1/3
        elif i==(n-1)*m:
            A[i,i-m]=1/3
            A[i,i-m+1]=1/3
            A[i,i+1]=1/3
        elif i==n*m-1:
            A[i,i-m-1]=1/3
            A[i,i-m]=1/3
            A[i,i-1]=1/3
        elif i<m:
            A[i,i-1]=1/5
            A[i,i+1]=1/5
            A[i,i+m-1]=1/5
            A[i,i+m]=1/5
            A[i,i+m+1]=1/5
        elif i>(n-1)*m:
            A[i,i-m-1]=1/5
            A[i,i-m]=1/5
            A[i,i-m+1]=1/5
            A[i,i-1]=1/5
            A[i,i+1]=1/5
        elif i%m==0:
            A[i,i-m]=1/5
            A[i,i-m+1]=1/5
            A[i,i+1]=1/5
            A[i,i+m]=1/5
            A[i,i+m+1]=1/5
        elif i%m==(-1%m):
            A[i,i-m-1]=1/5
            A[i,i-m]=1/5
            A[i,i-1]=1/5
            A[i,i+m-1]=1/5
            A[i,i+m]=1/5
        else:
            A[i,i-m-1]=1/8
            A[i,i-m]=1/8
            A[i,i-m+1]=1/8
            A[i,i-1]=1/8
            A[i,i+1]=1/8
            A[i,i+m-1]=1/8
            A[i,i+m]=1/8
            A[i,i+m+1]=1/8
    return A

def soften(Z,it=2,ep=None):

    pad = padMatrix(Z.shape[0],Z.shape[1])
    if ep:
        err=ep+1
        while err>ep:
            Zp=Z
            Z=pad.dot(np.ravel(Z)).reshape(Z.shape)
            err=(Z-Zp).max()
            print(err)
        return Z
    else:
        for i in range(it):
            Z=pad.dot(np.ravel(Z)).reshape(Z.shape)
        return Z

def sigmoide(cells,mean=None):
    if mean is None:
        mean = np.mean(cells,0)

    sig = (cells-mean)>0

    sig = sig*2-1

    return sig

def energy(cells,W):
    E = -0.5*np.multiply(cells,np.dot(cells,W)).sum(1)
    return E


def create_landscape(adata,tag,cluster,clustcol,embedding=None,lims=None,res=None,copy = False,update=False):

    if f'Landscapes:{tag}' in adata.uns.keys() and cluster in adata.uns['Landscapes:'+tag].keys() and not copy and not update:
        print(f'Landscape {tag}:{cluster} already exists...')
        return

    genes = adata.uns[tag][cluster].index.tolist()
    cell_idx = adata.obs[clustcol]==cluster

    cells = pd.DataFrame(adata.layers["spliced"][cell_idx, :].todense(), columns=adata.var.index).loc[:,genes]

    W = adata.uns[tag][cluster]

    if embedding is None:
        print('No embedding selected. Creating PCA with the cluster cells')
        embedding = skd.PCA(n_components=2).fit(cells)


    ss = sigmoide(cells)

    p_cells = embedding.transform(cells)

    cells_E = energy(sigmoide(embedding.inverse_transform(p_cells),mean=np.mean(cells.to_numpy(),0)),W)

    minx,miny = np.min(p_cells,axis=0)
    maxx,maxy = np.max(p_cells,axis=0)

    cx,dx = ((maxx+minx)/2,(maxx-minx)/2)
    cy,dy = ((maxy+miny)/2,(maxy-miny)/2)

    lims = lims if lims is not None else [cx-1.05*dx,cx+1.05*dx,cy-1.05*dy,cy+1.05*dy]
    res = res if res is not None else 50

    X,Y = np.mgrid[lims[0]:lims[1]:complex(0,res),lims[2]:lims[3]:complex(0,res)]

    mesh = embedding.inverse_transform(np.array([np.ravel(X),np.ravel(Y)]).T)
    meshb = sigmoide(mesh,np.mean(cells.to_numpy(),0,keepdims=True))
    meshE = np.reshape(energy(meshb,W),X.shape)
    meshEsoft = soften(meshE)


    ls = {'X':X,'Y':Y,'E':meshEsoft,'cells_X':p_cells[:,0].T,'cells_Y':p_cells[:,1].T,'cells_E':cells_E}

    if copy == False:

        if f'Landscapes:{tag}' in adata.uns.keys():
            adata.uns[f'Landscapes:{tag}'][cluster] = ls
            print(f"--> Updated adata.uns with key 'Landscapes:{tag}:{cluster}'")
        else:
            adata.uns[f'Landscapes:{tag}'] = dict()
            adata.uns[f'Landscapes:{tag}'][cluster] = ls
            print(f"--> Updated adata.uns with key 'Landscapes:{tag}:{cluster}'")

    else:
        return ls


def create_all_landscapes(adata,tag,clustcol,embedding=None,lims=None,res=None,copy = False,update=False):

    n_clusts = np.size(list(set(adata.obs[clustcol])))

    if lims is None:
        lims = [None]*n_clusts

    elif isinstance(lims[0],int):
        lims = [lims]*n_clusts

    if copy:
        clusts = []
        for i,cluster in enumerate(list(set(adata.obs[clustcol]))):
            clusts.append(create_landscape(adata,tag,cluster,clustcol,embedding,lims[i],res,copy))

        clusts = np.array(clusts)

        return clusts

    for i,cluster in enumerate(list(set(adata.obs[clustcol]))):
        create_landscape(adata,tag,cluster,clustcol,embedding,lims[i],res,copy,update)


def plot_landscape(adata,tag,cluster,plot_cells=False,fig=None,ax=None,**kwargs):

    fig = fig if fig is not None else plt.figure()
    ax = ax if ax is not None else plt.axes(projection='3d')

    ls = adata.uns['Landscapes:'+tag][cluster]

    X = ls['X']
    Y = ls['Y']
    Z = ls['E']

    cmap = kwargs.pop('cmap') if 'cmap' in kwargs else colors.Greys

    ax.plot_surface(X,Y,Z,alpha=0.5,cmap=cmap)

    if plot_cells:
        Xc = ls['cells_X']
        Yc = ls['cells_Y']
        Zc = ls['cells_E']
        ax.scatter(Xc,Yc,Zc)
