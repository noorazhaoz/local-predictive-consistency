import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

rng = np.random.default_rng(3)

# ---- create clusters manually ----
def blob(center, n, scale=0.12):
    return center + scale*rng.standard_normal((n,2))

X = np.vstack([
    blob(np.array([0,0]), 22),          # center harmonic cluster
    blob(np.array([1.1,0.9]), 18),      # neighbourhood 1
    blob(np.array([-1.2,0.8]), 18),     # neighbourhood 2
    blob(np.array([0.7,-1.1]), 18)      # neighbourhood 3
])
n = len(X)

# groups just for colour
g = rng.choice([0,1], size=n, p=[0.65,0.35])

# kNN graph
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X)
dist, ind = nbrs.kneighbors(X)

edges = set()
for i in range(n):
    for j in ind[i,1:]:
        edges.add(tuple(sorted((i,int(j)))))

# predictions: smooth + a few local anomalies
f = 0.6*X[:,0] - 0.2*X[:,1] + 0.05*rng.standard_normal(n)

# inject anomalies only in outer clusters
bad_idx = rng.choice(range(22,n), size=4, replace=False)
f[bad_idx] += 1.4

# build simple S from adjacency
W = np.zeros((n,n))
for i,j in edges:
    W[i,j]=W[j,i]=1
D = W.sum(1,keepdims=True)+1e-9
S = W/D

r = np.abs(f - S@f)

# edges to highlight: only those touching large residual nodes
hi = set(np.argsort(r)[-6:])
red = [e for e in edges if e[0] in hi or e[1] in hi]
gray = [e for e in edges if e not in red]

# ---- plot ----
fig,ax = plt.subplots(figsize=(7,7))

# edges
for i,j in gray:
    ax.plot([X[i,0],X[j,0]],[X[i,1],X[j,1]],c="gray",lw=1,alpha=.5)
for i,j in red:
    ax.plot([X[i,0],X[j,0]],[X[i,1],X[j,1]],c="red",lw=2.5)

# nodes
sizes = 80 + 240*(r-r.min())/(r.max()-r.min()+1e-9)
for i in range(n):
    c = "#2b6cb0" if g[i]==0 else "#2f855a"
    ax.scatter(X[i,0],X[i,1],s=sizes[i],color=c,edgecolors="white",zorder=3)

# circles showing locality scale
for rad,col,ls in [(0.6,"#2f855a","--"),(1.5,"gray","-"),(2.0,"#2b6cb0","--")]:
    ax.add_patch(plt.Circle((0,0),rad,fill=False,linewidth=3,color=col,linestyle=ls,alpha=.7))

ax.set_title("Feature space $X$",fontsize=20)
ax.axis("equal")
ax.axis("off")
plt.show()