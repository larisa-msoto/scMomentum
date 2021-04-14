import matplotlib.cm as colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition as skd


def padMatrix(n,m):
    A= 0.125*np.ones(n,m)
    
    for i in range(min(n,m)):
        A[0,i]   = 0.2
        A[n-1,i] = 0.2
        A[i,0]   = 0.2
        A[i,m-1] = 0.2
        
    if n>m:    
        for i in range(m,n):
            A[i,0]   = 0.2
            A[i,m-1] = 0.2
        
    elif m>n:
        for i in range(n,m):
            A[0,i]   = 0.2
            A[n-1,i] = 0.2        
    
    
    A[0,0]     = 1/3
    A[0,m-1]   = 1/3
    A[n-1,0]   = 1/3
    A[n-1,m-1] = 1/3
    
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


def create_landscape(adata,tag,cluster,clustcol,embedding=None,lims=None,res=None,copy = False):
    
    genes = adata.uns[f'{tag}:{cluster}'].index.tolist()
    cell_idx = adata.obs[clustcol]==cluster
    
    cells = pd.DataFrame(adata.layers["spliced"][cell_idx, :].todense(), columns=adata.var.index).loc(:,genes)
    
    W = adata.uns[f'{tag}:{cluster}']

    if embedding is None:
        print('No embedding selected. Creating PCA with the cluster cells')
        embedding = skd.PCA(n_components=2).fit(cells)
        
    
    ss = sigmoide(cells)
    cells_E = energy(ss,W)
    
    p_cells = embedding.transform(cells)
    
    minx,miny = np.min(p_cells,axis=0)
    maxx,maxy = np.max(p_cells,axis=0)
    
    cx,dx = ((maxx+minx)/2,(maxx-minx)/2)
    cy,dy = ((maxy+miny)/2,(maxy-miny)/2)
    
    lims = lims if lims is not None else [cx-1.05*dx,cx+1.05*dx,cy-1.05*dy,cy+1.05*dy]
    
    X,Y = np.mgrid[lims[0]:lims[1]:complex(0,res),lims[2]:lims[3]:complex(0,res)]
    
    mesh = embedding.inverse_transform(np.array([np.ravel(X),np.ravel(Y)]).T)
    meshb = sigmoide(mesh,np.mean(cells,0))
    meshE = np.reshape(energy(meshb,W),X.shape)
    meshEsoft = soften(meshE)
    
    
    ls = {'X':X,'Y':Y,'E':E,'cells_X':p_cells[:,0].T,'cells_Y':p_cells[:,1].T,'cells_E':cells_E}
    
    if copy == False:

        if f'Landscapes:{tag}' in adata.uns.keys():
            adata.uns[f'Lanscapes:{tag}'][cluster] = ls
            print(f"--> Updated adata.uns with key 'Landscape:{tag}:{cluster}'")
        else:
            adata.uns[f'Landscapes:{tag}'] = dict()
            adata.uns[f'Landscapes:{tag}'][cluster] = ls
            print(f"--> Updated adata.uns with key 'Landscape:{tag}:{cluster}'")

    else:
        return ls
        

def create_all_landscapes(adata,tag,clustcol,embedding=None,lims=None,res=None,copy = False):
    
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
        create_landscape(adata,tag,cluster,clustcol,embedding,lims[i],res,copy)
        

def plot_landscape(adata,tag,cluster,plot_cells=False,fig=None,ax=None,**kwargs):
    
    fig = fig if fig is not None else plt.figure()
    ax = ax if ax is not None else fig.subplots(projection='3d')
    
    ls = adata.uns[f'Landscapes:{tag}:{cluster}']
    
    X = ls['X']
    Y = ls['Y']
    Z = ls['E']
    
    cmap = kwargs.pop['cmap'] if 'cmap' in kwargs else colors.Greys
    
    ax.plot_surface(X,Y,Z,alpha=0.5,cmap=cmap)
    
    if plot_cells:
        Xc = ls['cells_X']
        Yc = ls['cells_Y']
        Zc = ls['cells_E']
        ax.scatter(Xc,Yc,Zc)
    
    