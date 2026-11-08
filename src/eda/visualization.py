"""Exported figures always use measured data."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .loaders import ROOT

plt.rcParams.update({'figure.dpi':120, 'axes.spines.top':False, 'axes.spines.right':False})


def save(name):
    path=ROOT / 'reports/figures' / (name+'.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return str(path)


def overview(frame):
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    counts=frame.source_type.value_counts().sort_index()
    axes[0].bar(counts.index,counts.values)
    axes[0].set(title='Corpus size',ylabel='Documents')
    sources=sorted(frame.source_type.unique())
    axes[1].boxplot([frame.loc[frame.source_type==s,'tokens'] for s in sources],tick_labels=sources)
    axes[1].set(yscale='log',title='WordPiece lengths',ylabel='Tokens per document (log)')
    means=frame.groupby('source_type').code_fraction.mean()
    axes[2].bar(means.index,means.values*100)
    axes[2].set(title='Fenced code share',ylabel='Mean characters in code (%)')
    for ax in axes:
        ax.tick_params(axis='x',labelrotation=30)
    save('corpus_overview')


def heatmap(frame,name,title):
    plt.figure(figsize=(max(7,len(frame.columns)*.8),max(3.5,len(frame.index)*.7)))
    plt.imshow(frame.to_numpy(dtype=float),cmap='Blues',vmin=0,vmax=1,aspect='auto')
    plt.xticks(range(len(frame.columns)),frame.columns,rotation=30,ha='right')
    plt.yticks(range(len(frame.index)),frame.index)
    for i in range(len(frame.index)):
        for j in range(len(frame.columns)):
            plt.text(j,i,f'{float(frame.iloc[i,j]):.2f}',ha='center',va='center',color='white' if float(frame.iloc[i,j])>.55 else 'black')
    plt.colorbar(label='Ratio'); plt.title(title); save(name)


def semantic_plot(pca, sources, pairs):
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for source in sorted(set(sources)):
        ix=np.array(sources)==source
        axes[0].scatter(pca[ix,0],pca[ix,1],label=source,alpha=.65,s=18)
    axes[0].legend(); axes[0].set(title='PCA of document-level mean chunk embeddings',xlabel='PC1',ylabel='PC2')
    for group in ['within','cross']:
        axes[1].hist(pairs.loc[pairs.group==group,'cosine'],bins=30,alpha=.5,density=True,label=group)
    axes[1].legend(); axes[1].set(title='Distinct document pair similarities',xlabel='Cosine similarity',ylabel='Density')
    save('semantic_space')
