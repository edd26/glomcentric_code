# %% markdown
# # Hierachical cluster odor spectra across animals
# %% codecell
%pylab inline
# %% codecell
import sys
import os
toplevelpath = os.path.realpath(os.path.pardir)

# %% codecell
import numpy as np
import matplotlib.pyplot as plt
import glob, csv, pickle, os, json
import matplotlib, scipy
from collections import defaultdict, OrderedDict
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage,dendrogram
from matplotlib import gridspec

from regnmf import ImageAnalysisComponents as ia
# %% codecell
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %% markdown
# ## Prepare Analysis
# %% markdown
# #### Specify parameter
# %% codecell
method = 'nnmf_150_sm2_convex_sp*_ios_meas' #'sica_200_ios_meas' #
animals =  ['111210sph', '111221sph','111222sph', '120107', '120119', '120121', '120125']
stimulusdrive = 0.4 # maximale trial2trial correlation distance (aka 1-correlation) of modes to be included
min_activation = 0.2 # maximale activation strength of modes to be included

datapath = os.path.join(toplevelpath, 'glomcentric_data_preprocessed')
factorization_path = os.path.join(datapath, 'MOBdecomposed')
bg_path = os.path.join(datapath, 'MOBconverted')
cas2name_file = os.path.join(datapath, 'DataDicts', 'Name2MomCas.tab')
cluster_file = os.path.join(datapath, 'DataDicts', 'cluster_assignment.json')
spec_savepath = os.path.join(datapath, 'DataDicts')
vis_path = os.path.join(datapath, 'Vis')
clustering_savepath = os.path.join(vis_path, 'Clustering')

# %% markdown
# #### Define Functions to process hierachichal cluster (linkages)
# %% codecell
def return_all_childs(mylinkage, parent):
    '''
    Recursive function returns all leafs of parent node in mylinkage.
    A leaf corresponds to the index of the object in the timeseries.'''
    num_leaves = mylinkage.shape[0]+1
    parent = int(parent)
    if parent<num_leaves:
        return [parent]
    else:
        leftchild, rightchild = mylinkage[parent-num_leaves][:2]
        return return_all_childs(mylinkage, parent=leftchild) + return_all_childs(mylinkage, parent=rightchild)

def return_all_links(mylinkage, parent):
    ''' recursive function returns all links of parent node in mylinkage'''
    num_leaves = mylinkage.shape[0]+1
    parent = int(parent)
    if parent<num_leaves:
        return [parent]
    else:
        leftchild, rightchild = mylinkage[parent-num_leaves][:2]
        return [parent] + return_all_links(mylinkage, parent=leftchild) + return_all_links(mylinkage, parent=rightchild)

def color_clusters(cluster, color_dict = None, chex=False):
    ''' creates link-coloring function to color each cluster (given by parent node) '''
    cluster_colors = defaultdict(lambda: '0.5')
    for clust_ix, cluster_parent in enumerate(cluster):
        clust_color = color_dict[cluster_parent] if color_dict else plt.cm.prism(1.*clust_ix/len(cluster))
        if not(chex):
            clust_color = matplotlib.colors.rgb2hex(clust_color)
        colordict_update = {i: clust_color for i in return_all_links(link, cluster_parent+link.shape[0]+1)}
        cluster_colors.update(colordict_update)
    return lambda node: cluster_colors[node]
# %% markdown
# #### Define function to create combined timeseries of all animals
# %% codecell
def load_combined_series(allIDs, filemask, thres, min_strength, factorization_path):
    ''' function to load and preprocess timeseries of multiple animals '''

    allgood, turn = [], []
    for measID in allIDs:

        # load timeseries
        ts = ia.TimeSeries()
        filename = glob.glob(os.path.join(factorization_path, measID, filemask+'.npy'))
        assert len(filename)==1
        ts.load(filename[0].split('.')[0])
        ts.label_stimuli = [i.split('_')[0] for i in ts.label_stimuli]
        if '_l_' in ts.name:
            turn.append(measID)
            ts.base.set_series(ts.base.shaped2D()[:,::-1])
        ts.name = measID

        # calc odor spectrum of modes
        signal = ia.TrialMean()(ia.CutOut((2, 5))(ts))
        # calc t2t correlation, exclude modes with t2t < thres
        mode_cor = ia.CalcStimulusDrive()(signal)
        signal = ia.SelectObjects()(signal, mode_cor._series.squeeze()<thres)

        # calc single odor response
        signal = ia.SingleSampleResponse()(signal)

        # selected only modes with maximal activation above min_strength
        strength = np.max(signal._series,0)
        signal = ia.SelectObjects()(signal, strength>min_strength)

        allgood.append(signal)

    allgood = ia.ObjectConcat(unequalsample=True)(allgood)
    return allgood, turn
# %% markdown
# #### Read in data
# %% codecell
cas2name = {l[0]:l[1] for l in csv.reader(open(cas2name_file),  delimiter='\t')}

ts, turn = load_combined_series(animals, method, stimulusdrive, min_activation, factorization_path)

# load background images, turn if left bulb
bg_dict = {measID: plt.imread(os.path.join(bg_path, measID, 'bg.png')) for measID in animals}
for ani in turn:
    bg_dict[ani] = bg_dict[ani][::-1]
# %% markdown
# #### Sort odors by similarity
# %% markdown
# Define order (according to a hierachical clustering) in which odors are displayed. Shall provide a more intuitive reading of odor-spectra with nearby odors having similar glomerular activation profiles
# %% codecell
odor_link = linkage(ts._series, metric = 'correlation', method = 'average')

odor_d = dendrogram(odor_link, distance_sort='descending', no_plot=True)
odor_order = np.array(odor_d['leaves'])

ts = ia.SelectTrials()(ts, odor_order)
# %% markdown
# ## Hierachical cluster data
# %% markdown
# Plots a hierachical clustering. Colors cluster as defined in the file specified in cluster_file. Please add new cluster manually to cluster_file.
# %% codecell
load_cluster = True #if already cluster assignments exist, load them
cluster_metric = 'correlation'
linkage_scheme = 'average'

title = '_'.join((cluster_metric, linkage_scheme, method))
cluster = json.load(open(cluster_file))[title] if load_cluster else None
# create copy of timeseries with timecourse of unit length (does not influence correlation/cosine but euclidean distance)
ts_normed = ts.copy()
norm = np.sqrt(np.sum((ts._series**2),0))
ts_normed._series /= norm

# create hierachical clustering
link = linkage(ts_normed._series.T, metric = cluster_metric, method = linkage_scheme)
# %% markdown
# ##### About the data structure
# %% codecell
array = ts_normed.shaped2D()
animals = unique([s.split('_')[0] for s in ts_normed.label_objects])
glom_per_animal = {a: [o.split('_')[1] for o in ts_normed.label_objects if o.split('_')[0] == a] for a in animals}
print("ts_normed has {} stimuli and {} objects".format(len(ts_normed.label_stimuli),len(ts_normed.label_objects)))
print("The 2D array has shape {}, therefore stimuli on axis 0 and objects on axis 1".format(array.shape))
print("The 'objects' are glomeruli extracted from {} animals".format(len(animals)))
print("each animal has the following number of glomeruli:")
for k,i in glom_per_animal.items():
    print("{}: {} gloms".format(k, len(i)))
# %% markdown
# ## Plot location and spectra of cluster
# %% markdown
# #### Functions to plot collection of spectra
# %% codecell
def allspec_plot(ax, spectra, color):
   ''' plots all spectra individual'''
   ax.plot(spectra, color=color)
   ax.yaxis.get_major_locator().set_params(nbins=3, prune='both')

def percentile_plot(ax, spectra, color):
    '''plots median, quartiles and min/max of data'''
    ax.plot(np.median(spectra,1), lw=2, color=color)
    for low, high in [(0,100), (25,75)]:
        ax.fill_between(range(spectra.shape[0]), np.percentile(spectra,low,axis=1), np.percentile(spectra,high,axis=1),
                    facecolor=color, alpha=0.25)
    ax.yaxis.get_major_locator().set_params(nbins=3, prune='both')

def percentile_plot2(ax, spectra, color):
    '''plots median, quartiles and min of data'''
    ax.plot(np.median(spectra,1), lw=2, color=color)
    for low, high in [(0,75), (25,75)]:
        ax.fill_between(range(spectra.shape[0]), np.percentile(spectra,low,axis=1), np.percentile(spectra,high,axis=1),
                    facecolor=color, alpha=0.25)
    ax.yaxis.get_major_locator().set_params(nbins=3, prune='both')

def mean_plot(ax, spectra, color):
    '''plots mean'''
    ax.plot(np.mean(spectra,1), lw=2, color=color)
    ax.yaxis.get_major_locator().set_params(nbins=3, prune='both')

def mean_plot_heatmap(ax, spectra, color):
    '''plots mean, quartiles and min of data'''
    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list('tmp', ['0',color])
    ax.imshow(np.mean(spectra,1).reshape((1,-1)), vmin=0, cmap=mycmap, aspect='auto', interpolation='none')
    ax.set_yticks([])

# %% markdown
# #### Functions to create cluster visualization
# %% codecell
def plot_location(ts,
                  link,
                  cluster,
                  bg_dict,
                  ax_dict,
                  color_dict,
                  face=True,
                  base_thres=0.3,
                  scalebar=True):
    ''' plot cluster location'''

    # plot bg
    for axname,ax in ax_dict.items():
        bg = bg_dict[axname].copy()
        if scalebar:
            pixel_size = 1.63/1344. *1000. #µm
            len_200 = int(round(200./pixel_size))   # 200 µm
            bg[950:970,(300-len_200):300]=1.

        ax.imshow(bg, interpolation='none', cmap=plt.cm.bone, extent=[0,84,64,0])
        ax.set_axis_off()

    # plot location and timecouses of clusters
    for clust in cluster:
        color = color_dict[clust]
        spec_collection = []
        for node in return_all_childs(link,clust+link.shape[0]+1):
            # get animal of cluster member
            measID = ts.label_objects[node].split('_')[0]
            # print pixel participation
            mode = ts.base.shaped2D()[node]
            if face:
                ax_dict[measID].contourf(mode, [base_thres,1], colors=[color], alpha=0.5)
            else:
                ax_dict[measID].contour(mode, [base_thres], colors=[color], alpha=0.5, linewidths=[2])

    #plot A-P arrows in the last plot
    scalefac = 1.63/1344.
    ap_arrowcolor = [1.,1.,1.]
    arr1 = ax.arrow(82,60,-8,0, head_width=2.5, color=ap_arrowcolor, linewidth=0.65, zorder=10)
    arr2 = ax.arrow(82,60,0,-8, head_width=2.5, color=ap_arrowcolor, linewidth=0.65, zorder=10)
    arrlabA = ax.text(69,60, 'A', color=ap_arrowcolor, fontsize=6, ha='right', va='center', zorder=10)
    arrlabA = ax.text(81,41, 'L', color=ap_arrowcolor, fontsize=6, ha='center', va='top', zorder=10)



def plot_spec(ts, link, cluster, ax_dict, color_dict, plot_spec_func=allspec_plot, norm=[]):
    '''
    plot cluster spectra.

    ts - timeseries object
    link - linkage that defines which nodes are part of the cluster to plot
    cluster - index of the cluster for which to plot the spectra
    ax_dict - axes dictionary
    color_dict - specifying colors for the plots
    '''

    for clust in cluster:
        # get data
        color = color_dict[clust]
        nodes = return_all_childs(link,clust+link.shape[0]+1)
        spec_collection = [ts.matrix_shaped()[:,node] for node in nodes]

        # plot spectrum
        plot_spec_func(ax_dict[clust], np.array(spec_collection).T, color)
        ax_dict[clust].set_xlim([-0.5, num_stim-0.5])

        if len(norm) > 0:
            # note response strength
            ax_dict[clust].text(0.9,0.8, 'norm: %.1f'%np.mean(norm[nodes]),  transform=ax_dict[clust].transAxes)
# %% markdown
# #### Calc cluster prototype spectra
# %% codecell
condense = np.median #how to calc prototype

spec, bases =[], []
for clust in cluster:
    # calc cluster spec
    nodes = return_all_childs(link,clust+link.shape[0]+1)
    spec_collection = np.array([ts_normed.matrix_shaped()[:,node] for node in nodes])
    spec.append(condense(spec_collection,0))
    # create cluster base
    temp_dict = defaultdict(list)
    for node in nodes:
        animal = ts_normed.label_objects[node].split('_')[0]
        temp_dict[animal].append(node)
    base = []
    for animal in animals:
        if animal in temp_dict:
            base.append(np.sum(ts_normed.base._series[temp_dict[animal]],0))
        else:
            base.append(np.zeros(ts_normed.base._series[0].shape))
    bases.append(np.hstack(base))

# cast into timeseries object
ts_glom = ts_normed.copy()
ts_glom.label_objects = ['clust_%d'%clust for clust in cluster]
ts_glom.name = title
ts_glom._series = np.array(spec).T
ts_glom.base._series = np.vstack(bases)
ts_glom.base.typ.append('multiple')
ts_glom.base.shape = [ts_normed.base.shape]*len(animals)
# %% markdown
# #### calculate correlation to MOR18-2 cluster
# %% codecell
cluster_id = 200
metric = 'correlation'
what = ts_normed

protoype_idx = return_all_childs(link,cluster_id+link.shape[0]+1)
prototype = np.mean(what._series[:,protoype_idx],1)
cor = squareform(pdist(np.hstack([prototype[:,np.newaxis], what._series]).T, metric))[0,1:]
# %% markdown
# #### Functions to compute intra vs inter cluster distances (robustness of linkage)
# %% codecell
def inter_intra_dist(ts, link, cluster_id, metric='correlation'):
    ''' distance between cluster modes and distances of cluster modes to remaining modes'''

    member_idx = return_all_childs(link,cluster_id+link.shape[0]+1)
    member_mask = np.zeros(ts.num_objects).astype('bool')
    member_mask[member_idx] = True
    dist = pdist(ts._series.T, metric)
    dist_inter = squareform(dist)[member_mask][:,np.logical_not(member_mask)]
    dist_intra = squareform(squareform(dist)[member_mask][:,member_mask])
    return dist_inter, dist_intra

def inter_intra_prototypedist(ts, link, cluster_id, metric='correlation', prototyping=np.mean):
    ''' distance of cluster protoype (average cluster spectrum) to all cluster modes and to remaining modes'''

    member_idx = return_all_childs(link,cluster_id+link.shape[0]+1)
    member_mask = np.zeros(ts.num_objects).astype('bool')
    member_mask[member_idx] = True
    prototype = np.mean(ts_normed._series[:,member_idx],1)
    dist = squareform(pdist(np.hstack([prototype[:,np.newaxis], ts._series]).T, metric))[0,1:]
    dist_inter = dist[np.logical_not(member_mask)]
    dist_intra = dist[member_mask]
    return dist_inter, dist_intra
# %% markdown
# #### Basic layout definition
# %% codecell
fig_dim = (7.48,9.4)
global_fs= 7

layout = {   'axes.labelsize': 7,
             'axes.linewidth': .5,
             'xtick.major.size': 2,     # major tick size in points
             'xtick.minor.size': 1,     # minor tick size in points
             'xtick.labelsize': 7,       # fontsize of the tick labels
             'xtick.major.pad': 2,

             'ytick.major.size': 2,      # major tick size in points
             'ytick.minor.size': 1,     # minor tick size in points
             'ytick.labelsize':7,       # fontsize of the tick labels
             'ytick.major.pad': 2,

             'mathtext.default' : 'regular',
             'legend.fontsize': 7,
             'figure.dpi':150
             }

import matplotlib as mpl
for k, v in layout.items():
    mpl.rcParams[k] = v
# %% markdown
# ### Figure: Fingerprinting (MOR18-2 and other examples)
# %% codecell
cluster = [200, 172, 256, 254, 278]
num_cluster = len(cluster)

fig = plt.figure(figsize=fig_dim) #(15,1*(num_cluster+2)))
gs_meta = matplotlib.gridspec.GridSpec(3, 1, bottom=0.5, top = 0.99, left = 0.05, right=0.98,
                                       height_ratios=[1, 1.5, 0.9*num_cluster], hspace=0.07)
axlegend = fig.add_axes([0.8, 0.45, 0.02, 0.02])

metric = 'correlation'
num_animals = len(animals)
num_stim = len(ts.label_stimuli)
clust_colors = {clust: plt.cm.gist_rainbow(1.*cluster.index(clust)/num_cluster) for clust in cluster}

# plot dendrogram
ax = fig.add_subplot(gs_meta[0])
top = gs_meta[0].get_position(fig).corners()[1,1]
fig.text(0.005, top, '(a)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
lw = matplotlib.rcParams['lines.linewidth']
matplotlib.rcParams['lines.linewidth'] = 0.6
d = dendrogram(link, link_color_func = color_clusters(cluster, clust_colors), count_sort='descending')
matplotlib.rcParams['lines.linewidth'] = lw
ax.set_xticks([])
ax.set_yticks([0,0.4,1])
ax.set_ylabel('$\hat{d}_{r}$', labelpad=-1)
ax.set_xlabel('glomeruli', labelpad=3)

# plot locations
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, num_animals, gs_meta[1], wspace=0.01)
top = gs_meta[1].get_position(fig).corners()[1,1]
fig.text(0.005, top, '(b)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
axbase = {animal: fig.add_subplot(gs[0,ix]) for ix, animal in enumerate(animals)}
ax = plot_location(ts_normed, link, cluster, bg_dict, axbase, clust_colors, face=False)
#plot_location(ts_normed, link, [cluster[0]], bg_dict, axbase, clust_colors)




# prepare axes for plotting spectra and cluster distances
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(num_cluster, 3, gs_meta[2], hspace=0.15,
                                                 width_ratios=[6,1,1], wspace=0.2)
# create panel labels
top = gs_meta[2].get_position(fig).corners()[1,1]
fig.text(0.005, top+0.005, '(c)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
left = gs[1].get_position(fig).corners()[1,0]
fig.text(left-0.03, top+0.005, '(d)', fontweight='bold', fontsize=global_fs, ha='right', va='center')
left = gs[2].get_position(fig).corners()[1,0]
fig.text(left-0.03, top+0.005, '(e)', fontweight='bold', fontsize=global_fs, ha='right', va='center')


# plot spectra
axtime = OrderedDict([(clust,fig.add_subplot(gs[ix,0])) for ix, clust in enumerate(cluster)])
lw = matplotlib.rcParams['lines.linewidth']
matplotlib.rcParams['lines.linewidth'] = 0.6
plot_spec(ts_normed, link, cluster, axtime, clust_colors)
matplotlib.rcParams['lines.linewidth'] = lw
for ax in axtime.values():
    ax.set_yticks([0.2,0.4])
    ax.set_ylabel('$a_{norm}$', labelpad=0.5)
    ax.set_xticklabels([])
    ax.set_xticks(np.arange(num_stim), minor=True)
    ax.set_xticks(np.arange(0,num_stim, 5))
axtime.values()[-1].set_xticklabels([cas2name[i].decode('utf-8') for i in ts.label_stimuli],
                                        rotation='70', ha='right', minor=True)

# plot cluster separation histogramms
for ix, clust in enumerate(cluster):
    cor_inter, cor_intra = inter_intra_dist(ts_normed, link, clust, metric=metric)
    ax1 = fig.add_subplot(gs[ix,1])
    h = ax1.hist([1-cor_inter, [1-cor_intra]], np.linspace(-1,1,41), color=['0.5', clust_colors[clust]],
            log=True, histtype='barstacked', lw=0, rwidth=0.9, label=['inter', 'intra'])
    ax1.set_ylim((0.3,500))
    ax1.set_yticks([1,10,100])
    ax1.set_ylabel('#', labelpad=0.2)
    ax1.set_xticks([-1,-0.5,0,0.5,1])
    ax1.set_xticklabels([])
    [ax1.spines[i].set_color('none') for i in ['top', 'right']]
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')

    cor_inter, cor_intra = inter_intra_prototypedist(ts_normed, link, clust, metric=metric)
    ax2 = fig.add_subplot(gs[ix,2])
    ax2.hist([1-cor_inter, 1-cor_intra], np.linspace(-1,1,41), color=['0.5',clust_colors[clust]],
            log=True, histtype='barstacked', lw=0, rwidth=0.9)
    ax2.set_ylim((0.3,50))
    ax2.set_yticks([1,10])
    ax2.set_ylabel('#', labelpad=0.2)
    ax2.set_xticks([-1,-0.5,0,0.5,1])
    ax2.set_xticklabels([])
    [ax2.spines[i].set_color('none') for i in ['top', 'right']]
    ax2.yaxis.set_ticks_position('left')
    ax2.xaxis.set_ticks_position('bottom')

ax1.set_xticks([-1,-0.5,0,0.5,1])
ax1.set_xticklabels([-1,"",0,"",1])
ax1.set_xlabel('$r$',labelpad=1)
ax2.set_xticks([-1, -0.5,0,0.5,1])
ax2.set_xticklabels([-1, '', 0, '' ,1])
ax2.set_xlabel('$r^{centre}$', labelpad=0.3)

axlegend.set_axis_off()
[axlegend.plot([i,i+1], [0.05,0.05], color = clust_colors[c], lw=3) for i,c in enumerate(cluster)]
axlegend.plot([0,len(cluster)], [0,0], color = '0.5', lw=5)
axlegend.text(len(cluster)+1.5, 0.05, 'intra cluster', fontsize=global_fs, va='center')
axlegend.text(len(cluster)+1.5, 0, 'inter cluster', fontsize=global_fs, va='center')
axlegend.set_ylim((-0.02,0.07))


# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'tunotopy.png')
fig.savefig(savename, bbox_inches='tight', dpi=600)

# %% markdown
# #### Overview on cas numbers, odorant names and responses
# %% codecell
from pandas import DataFrame
cas = ts_normed.label_stimuli
names = [cas2name[n] for n in cas]
df = DataFrame(index=cas, columns=['names'])
df['names'] = names
for clust in cluster:
    nodes = return_all_childs(link,clust+link.shape[0]+1)
    spec_collection = [ts_normed.matrix_shaped()[:,node] for node in nodes]
    df[clust] = numpy.mean(np.array(spec_collection), axis=0)
df
# %% markdown
# ### Fig 5: Tunotopic Neighbours
# %% markdown
# #### nearest cluster
# %% codecell
mor182cor = squareform(pdist(ts_glom._series.T, 'correlation'))[ts_glom.label_objects.index('clust_200')]
order = np.argsort(mor182cor)
for i in order[:20]:
    print ts_glom.label_objects[i], mor182cor[i]
# %% markdown
# #### Locations (peaks) of glomeruli
# %% codecell
# spatial locations of glomeruli: peaks in the modes
from skimage.feature import peak_local_max
from itertools import compress
peaks = []
th = 0.7 #threshold for peak
for i in range(ts_normed.base.shaped2D().shape[0]):
    peak = peak_local_max(ts.base.shaped2D()[i,:,:], threshold_abs=th, min_distance=4, exclude_border=False)
    peaks.append(peak)
# %% codecell
#now need to compute a distance matrix between all peaks, using the closest peak in the cases where there are multiple.
from itertools import product
spatial_distmat = np.zeros((len(peaks), len(peaks)))
for i,p1 in enumerate(peaks):
    for j, p2 in enumerate(peaks):
        if (len(p1) < 1) or (len(p2) < 1):
            spatial_distmat[i,j] = np.nan
        else: #need to take into account multiple peaks
            spatial_distmat[i,j] = np.min([np.linalg.norm(pp2 - pp1) for pp1,pp2 in product(p1, p2)])

pixel_size = 1.63/1344.*8. *1000. #µm
spatial_distmat *= pixel_size
pixel_size
# %% markdown
# #### plot tunotopy figure
# %% codecell
cluster = [200, 292, 257, 228, 264, 183, 281, 220, 247, 319, 146, 254]
num_cluster = len(cluster)

fig = plt.figure(figsize=fig_dim)
gs_meta = matplotlib.gridspec.GridSpec(3, 1, bottom=0.63, top = 0.99, left = 0.05, right=0.99,
                                       height_ratios=[1,1.4,num_cluster*0.15], hspace=0.04)

num_animals = len(animals)
num_stim = len(ts.label_stimuli)
clust_colors = {clust: plt.cm.gist_rainbow_r(1.*cluster.index(clust)/num_cluster) for clust in cluster}

gs_top = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 3, gs_meta[0], hspace=0.01, height_ratios=[5,1],
                                                     wspace=0.05, width_ratios=[16.5,0.3,0.5])

# plot dendrogram
ax = fig.add_subplot(gs_top[0,0])
top = gs_meta[0].get_position(fig).corners()[1,1]
fig.text(0.005, top, '(a)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
lw = matplotlib.rcParams['lines.linewidth']
matplotlib.rcParams['lines.linewidth'] = 0.6
d = dendrogram(link, link_color_func = color_clusters(cluster, clust_colors), count_sort='descending')
matplotlib.rcParams['lines.linewidth'] = lw
ax.set_xticks([])
ax.set_yticks([0,0.4,1])
ax.set_ylabel('$\hat{d}_{r}$', labelpad=-1)

# plot MOR18-2 correlation
ax = fig.add_subplot(gs_top[1,0])
im = ax.imshow((1-cor[d['leaves']]).reshape((1,-1)), cmap=plt.cm.seismic,
               interpolation='none',
               aspect='auto', vmin=-1, vmax=1)
ax.set_yticks([])
ax.set_xticks([])
ax.set_xlabel('glomeruli', labelpad=2)
# colorbar
axbar = fig.add_subplot(gs_top[:,1])
cbar = plt.colorbar(im, cax=axbar)
cbar.set_ticks([-1,0,1])
cbar.set_label('$r_{MOR18-2}$', labelpad=-2)


# plot locations
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, num_animals, gs_meta[1,:], wspace=0.02)
top = gs_meta[1].get_position(fig).corners()[1,1]
fig.text(0.005, top-0.01, '(b)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
axbase = {animal: fig.add_subplot(gs[0,ix]) for ix, animal in enumerate(animals)}
plot_location(ts_normed, link, cluster, bg_dict, axbase, clust_colors)


# prepare axes for plotting spectra
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(num_cluster, 3, gs_meta[2], hspace=0, wspace=0.05,
                                                 width_ratios=[16.5,0.3,0.5])
top = gs_meta[2].get_position(fig).corners()[1,1]
fig.text(0.005, top, '(c)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
axtime = OrderedDict([(clust,fig.add_subplot(gs[ix,0])) for ix, clust in enumerate(cluster)])

# plot spectra
plot_spec(ts_normed, link, cluster, axtime, clust_colors, mean_plot_heatmap)
for ax in axtime.values():
    ax.set_xticklabels([])
    ax.set_xticks([])
ax.set_xticks(np.arange(num_stim)-0.3)
ax.set_xticklabels([cas2name[i].decode('utf-8') for i in ts.label_stimuli],
                                        rotation='70', ha='right')
ax.xaxis.set_tick_params(direction='out', top='off')
# colorbar
cbar = []
for i, clust in enumerate(cluster):

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('tmp', ['0',clust_colors[clust]])
    cbar.append([cmap(i) for i in np.arange(1,0,-0.01)])

axbar = fig.add_subplot(gs[:,1])
axbar.imshow(np.array(cbar).swapaxes(0,1), interpolation='none', aspect='auto')
axbar.set_xticks([])
axbar.set_yticks([0,100])
axbar.set_yticklabels(['max', '0'])
axbar.yaxis.set_ticks_position('right')
axbar.set_ylabel('a', labelpad=-9)
axbar.yaxis.set_label_position('right')




cluster1 = [200, 387, 391, 389, 394, 383, 400, 402][::-1] #2nd level neigbourhood
#cluster2 = [399, 400, 402] #3rd level neighbourhood

# define axes_layout
h_dend = 2 #heigth of dendrogramms
h_base = 2.5 #heigth of spatial plots
h_time = 0.25 #hieght of timeplots


clust_colors = {200:'#00FF00',
                387:'#33CC33',
                383:'#6699FF',
                391:'#0066FF',
                389:'#0033CC',
                394:'#0099FF',
                400:'#FF0000',
                402:'#AAAAAA'
                }

# plot first level cluster
dendrotop = 0.475
gs2 = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[0.6, 1], bottom=0.3, top = dendrotop,
                                        left = 0.05, right=0.99,  hspace=0.1)
fig.text(0.005, dendrotop, '(d)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
# plot dendrogram of first cluster
ax = fig.add_subplot(gs2[0])

lw = matplotlib.rcParams['lines.linewidth']
matplotlib.rcParams['lines.linewidth'] = 0.6
d = dendrogram(link, link_color_func = color_clusters(cluster1, clust_colors, chex=True), count_sort='descending')
matplotlib.rcParams['lines.linewidth'] = lw
ax.set_xticks([])
ax.set_xlabel('glomeruli', labelpad=2)
ax.set_yticks([0,0.4,1])
ax.set_ylabel('$\hat{d}_{r}$', labelpad=-1)


# plot locations
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, num_animals, gs2[1], wspace=0.02)
top = gs2[1].get_position(fig).corners()[1,1]
fig.text(0.005, top-0.01, '(e)', fontweight='bold', fontsize=global_fs, ha='left', va='center')
axbase = {animal: fig.add_subplot(gs[0,ix]) for ix, animal in enumerate(animals)}
plot_location(ts_normed, link, cluster1, bg_dict, axbase, clust_colors, base_thres=0.1)

# Revision: caluclate & plot centres of mass

clusters = {"MOR18-2":[200],
            "MOR18-2 patch":[387],
            "blue cluster":[383,391,389,394],
            "red cluster":[400],
            "grey cluster":[402] # (ignore)
           }
# first calculate per-animal:
# - locations of cluster members (in "peaks" data structure)
# - centres of mass of clusters
# - distances of centers of mass
# - directions (angles) of the lines between the centres of mass for each animal


animal_results = {}
for a in animals:
    locations = {}
    centers_of_mass = {}
    animal_mask = np.array([lo.startswith(a) for lo in ts_normed.label_objects])
    for clust_name in ["MOR18-2 patch", "blue cluster", "red cluster"]:
        members = []
        cluster_ids = clusters[clust_name]
        for c_id in cluster_ids:
            members.extend(return_all_childs(link, c_id+link.shape[0]+1))
        #filter for animals
        member_mask = np.zeros_like(animal_mask)
        member_mask[members] = True
        members = np.nonzero(np.logical_and(member_mask, animal_mask))[0]
        locations[clust_name] = []
        for m in members:
            locations[clust_name].extend(peaks[m])
        locations[clust_name] = np.array(locations[clust_name])
        centers_of_mass[clust_name] = np.mean(locations[clust_name], axis=0)
    animal_results[a] = {"centers_of_mass":centers_of_mass, "locations":locations}
    redcenter, bluecenter = [animal_results[a]["centers_of_mass"][i]
                             for i in ["red cluster","blue cluster"]]
    red_blue_vector = redcenter-bluecenter
    animal_results[a]["d_r_b"] = np.linalg.norm(red_blue_vector) * pixel_size
    if np.abs(red_blue_vector[0]) > 1e-5:
        angle = np.arctan(red_blue_vector[1]/red_blue_vector[0])
    else:
        angle = np.pi/2.
        if red_blue_vector[1] < 0.:
            angle *= -1.
    #full angle depends on quadrant
    if red_blue_vector[0]>=0.: #1st or 4th quadrant
        if red_blue_vector[1] < 0.: #4th quadrant
            angle += 2*np.pi
    else: #2nd or 3rd quadrant
        angle += np.pi
    animal_results[a]["a_r_b"] = angle
# add centres of mass to plot
for a in animals:
    ax = axbase[a]
    rpc = animal_results[a]["centers_of_mass"]["red cluster"]
    bpc = animal_results[a]["centers_of_mass"]["blue cluster"]
    gpc = animal_results[a]["centers_of_mass"]["MOR18-2 patch"]
    ax.plot(gpc[1], gpc[0], marker="x", color="lime")
    ax.plot(rpc[1], rpc[0], marker="x", color="orangered")
    ax.plot(bpc[1], bpc[0], marker="x", color="deepskyblue")
# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'tunotopy_neighborhood.png')
fig.savefig(savename, bbox_inches='tight', dpi=600)
# %% markdown
# #### Test for significance of patch glomeruli being closer to MOR18-2 than expected by chance
# %% codecell
from itertools import product
def intra_extra_patch_spatial_distance(ts, link, peaks, reference_cluster, patch_clusters):
    """
    Compute spatial distances between patch members to the closest MOR18-2, and patch non-members to MOR18-2.
    """
    ref_mask = np.zeros_like(ts.label_objects, dtype='bool')
    ref_members = return_all_childs(link, reference_cluster+link.shape[0]+1)
    ref_mask[ref_members] = True
    patch = []
    for p in patch_clusters:
        patch.extend(return_all_childs(link, p+link.shape[0]+1))
    patch_mask = np.zeros_like(ref_mask)
    patch_mask[patch] = True
    animal_list = np.array([lo.split("_")[0] for lo in ts.label_objects])
    animals = unique(animal_list)
    animal_dict = {}
    for a in animals:
        animal_dict[a] = animal_list == a
    d_patch = []
    d_nonpatch = []
    peaks = np.array(peaks)
    for animal,animal_mask in animal_dict.items():
        a_ref_mask = np.logical_and(animal_mask, ref_mask)
        a_patch_mask = np.logical_and(animal_mask, patch_mask)
        a_nonpatch_mask = np.logical_and(animal_mask,np.logical_not(np.logical_and(a_ref_mask, a_patch_mask)))
        ref_peaks = peaks[np.nonzero(a_ref_mask)[0]]
        patch_peaks = peaks[np.nonzero(a_patch_mask)[0]]
        nonpatch_peaks = peaks[np.nonzero(a_nonpatch_mask)[0]]
        for pp, rp in product(patch_peaks, ref_peaks):
            d_patch.append(np.min([np.linalg.norm(ppp - rpp) for ppp,rpp in product(pp, rp)]))
        for nop, rp in product(nonpatch_peaks, ref_peaks):
            d_nonpatch.append(np.min([np.linalg.norm(npp - rpp) for npp,rpp in product(nop, rp)]))
    return d_nonpatch, d_patch




# %% codecell
MOR182_cluster = 200
patch_clusters = [292, 257, 228, 264]
peaks_mum = np.array(peaks)* pixel_size
d_nonpatch, d_patch = intra_extra_patch_spatial_distance(ts_normed, link, peaks_mum, MOR182_cluster, patch_clusters)
v_mw, p_mw = scipy.stats.mannwhitneyu(d_patch, d_nonpatch, alternative="less")
v_mw_norm = v_mw/(len(d_nonpatch)* len(d_patch))
print("Odds: {:.3f}, p-value: {:.3g}".format(v_mw_norm, p_mw))

f = plt.figure(figsize=(3,2))
ax = f.add_subplot(111, frame_on=False)
bins = np.linspace(0, np.max(d_nonpatch), 20,endpoint=True)
spacing = bins[1]
heights, _ = np.histogram(d_nonpatch,bins)
ax.bar(bins[:-1], heights, width=spacing*.4, label="nonpatch")
heights, bins = np.histogram(d_patch, bins)
ax.bar(bins[:-1]+spacing*.5, heights, width=spacing*.4, label="patch")
ax.set_xlabel("distance [$\mu$m]")
ax.set_ylabel("frequency")
ax.legend(frameon=False)
# %% markdown
# Answer: Yes, patch glomeruli (as determined by hierarchical clustering on tuning similarities) are spatially more proximal to MOR18-2 than expected by chance (Mann-Whitney U test, $p<10^{-4}$, normalised U value=0.286). This means that when randomly drawing pairs of glomeruli from patch and non-patch clusters, the patch glomerulus will be closer to MOR18-2 than the non-patch glomerulus in about 71% of the cases. A control experiment where spatial distances does not yield a significant difference (normalised u: 0.531, p>0.7).
# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'patch_glom_distance_obs.svg')
f.savefig(savename)
# %% codecell
%%time
#shuffled_control

numrep = 10000
ustat = np.zeros((numrep, 2))
d_patch_shuf = np.array([])
d_nonpatch_shuf = np.array([])
for i in range(numrep):
    order = np.random.permutation(peaks_mum.shape[0])
    peaks_shuf = peaks_mum[order]
    dnp, dp = intra_extra_patch_spatial_distance(ts_normed, link, peaks_shuf, MOR182_cluster, patch_clusters)
    d_patch_shuf = np.concatenate((d_patch_shuf, dp))
    d_nonpatch_shuf = np.concatenate((d_nonpatch_shuf, dnp))
    v_mw, p_mw = scipy.stats.mannwhitneyu(dp, dnp, alternative="less")
    ustat[i,:] = [v_mw/(len(dnp)* len(dp)), p_mw]

print("Average odds: {:.3f}±{:.3g}, min p-value: {:.3g}±{:.3g}".format(np.mean(ustat[:,0]),
                                                          np.std(ustat[:,0]),
                                                          np.min(ustat[:,1]),
                                                          np.std(ustat[:,1])))



# %% codecell
f = plt.figure(figsize=(3,2))
ax = f.add_subplot(111, frame_on=False)
bins = np.linspace(0, np.max(d_nonpatch), 20,endpoint=True)
spacing = bins[1]
heights, _ = np.histogram(d_nonpatch_shuf,bins)
ax.bar(bins[:-1], heights, width=spacing*.4, label="nonpatch")
heights, bins = np.histogram(d_patch_shuf, bins)
ax.bar(bins[:-1]+spacing*.5, heights, width=spacing*.4, label="patch")
ax.set_xlabel("distance [$\mu$m]")
ax.set_ylabel("frequency")
ax.legend(frameon=False)
# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'patch_glom_distance_shuf.svg')
f.savefig(savename)
# %% codecell
#U stats on shuffled data.
f = plt.figure(figsize=(3,2))
ax = f.add_subplot(111, frame_on=False)
bins = np.linspace(np.min(ustat[:,0]), np.max(ustat[:,0]), 20, endpoint=True)
spacing = bins[1] - bins[0]
heights, _ = np.histogram(ustat[:,0], bins)
ax = f.add_subplot(111, frame_on=False)
ax.bar(bins[:-1], heights, width=spacing*.5, label="shuffled")
ax.plot([v_mw_norm,v_mw_norm], ax.get_ylim(), 'r-', label="observed", lw=0.7)
ax.legend(frameon=False)
np.sum(ustat[:,0]<v_mw_norm)*1./numrep
ax.set_xlabel("MW U (normalised)")
ax.set_ylabel("frequency")
# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'patch_glom_distance_shuf_ustat.svg')
f.savefig(savename)
# %% markdown
# ### Supercluster position relative to MOR18-2
# %% codecell
def plot_animal_results(animal_results):
    f = figure(figsize=(10,1))
    gs = plt.GridSpec(1,6)
    for g,a in zip(gs,animals):
        ax = f.add_subplot(g, frame_on=False, aspect=1.)
        rp = animal_results[a]["locations"]["red cluster"]
        rpc = animal_results[a]["centers_of_mass"]["red cluster"]
        bp = animal_results[a]["locations"]["blue cluster"]
        bpc = animal_results[a]["centers_of_mass"]["blue cluster"]
        gp = animal_results[a]["locations"]["MOR18-2 patch"]
        gpc = animal_results[a]["centers_of_mass"]["MOR18-2 patch"]
        ax.scatter(rp[:,1], rp[:,0], color="red")
        ax.scatter(bp[:,1], bp[:,0], color="blue")
        ax.scatter(gp[:,1], gp[:,0], color="green")
        ax.plot(rpc[1], rpc[0], marker="o", color="red", mec='k')
        ax.plot(bpc[1], bpc[0], marker="o", color="blue", mec='k')
        ax.plot(gpc[1], gpc[0], marker="o", color="green", mec='k')
        ax.set(**{"ylim":(64,0), "xlim":(0,84)})


# %% codecell
#distance between center of mass of blue and red: significant from shuffle control?
np.random.seed(2345234)
shuffles = []
num_shuffles = 10000
for n in range(num_shuffles):
    shuffled_animal_results = {}
    for a in animals:
        locations = animal_results[a]['locations']
        flat_locations = np.concatenate((locations['blue cluster'], locations['red cluster']))
        np.random.shuffle(flat_locations)
        shuffled_locations = {}
        for k,v in locations.items():
            if k == 'MOR18-2 patch':
                shuffled_locations[k] = locations[k]
            else:
                shuffled_locations[k] = flat_locations[:len(v)]
                flat_locations = flat_locations[len(v):]
        shuffled_animal_results[a] = {}
        shuffled_animal_results[a]['locations'] = shuffled_locations
        shuffled_animal_results[a]['centers_of_mass'] = {}
        for k,v in shuffled_locations.items():
            shuffled_animal_results[a]['centers_of_mass'][k] = np.mean(v, axis=0)
        redcenter, bluecenter = [shuffled_animal_results[a]["centers_of_mass"][i]
                                 for i in ["red cluster","blue cluster"]]
        red_blue_vector = redcenter-bluecenter
        shuffled_animal_results[a]["d_r_b"] = np.linalg.norm(red_blue_vector) * pixel_size
        if np.abs(red_blue_vector[0]) > 1e-5:
            angle = np.arctan(red_blue_vector[1]/red_blue_vector[0])
        else:
            angle = np.pi/2.
            if red_blue_vector[1] < 0.:
                angle *= -1.
        #full angle depends on quadrant
        if red_blue_vector[0]>=0.: #1st or 4th quadrant
            if red_blue_vector[1] < 0.: #4th quadrant
                angle += 2*np.pi
        else: #2nd or 3rd quadrant
            angle += np.pi
        shuffled_animal_results[a]["a_r_b"] = angle
    shuffles.append(shuffled_animal_results)
# %% codecell
plot_animal_results(animal_results)
plot_animal_results(shuffles[0])
# %% codecell
angles = []
for a in animals:
    angles.append(animal_results[a]["a_r_b"])
    print("animal {}: d:{:.2f} angle:{:.2f}; shuffled: avg d: {:.2f}".format(a,
                                          animal_results[a]["d_r_b"],
                                          animal_results[a]["a_r_b"],
                                          np.mean([s[a]["d_r_b"] for s in shuffles])))
angles = np.array(angles)

# %% codecell
angles_degrees = angles/(2*np.pi)*360
angles_degrees
# %% codecell
overall_angle = 90. + 360.- 320.05378444
print("angle spread in degrees: {}".format(overall_angle))
no_outl = 340.12312781 - 320.05378444
print("angle spread in degrees (without outlier): {}".format(no_outl))
# %% codecell
# histogram of distances
#exclude outlier
animals_no_outl = [a for a in animals if a != "120119"]
#shuffled_dists = np.array([s[a]["d_r_b"] for a in animals_no_outl for s in shuffles])
#animal_dists = np.array([animal_results[a]["d_r_b"] for a in animals_no_outl])
#with outlier
shuffled_dists = np.array([s[a]["d_r_b"] for a in animals for s in shuffles])
animal_dists = np.array([animal_results[a]["d_r_b"] for a in animals])
#fraction of shuffled distances larger than the average observed distance, and per animal
avg_animal_dist = np.mean(animal_dists)
frac_larger_shuf = np.sum(shuffled_dists>avg_animal_dist)/(1.*len(shuffled_dists))
print("fraction of distances from shufflev "\
      +" controls that are larger than the"\
      +" average observed value: {:.4f}".format(frac_larger_shuf))
#per animal:
#for a in animals_no_outl:
for a in animals:
    frac_larger_shuf = np.sum(shuffled_dists>animal_results[a]["d_r_b"])/(1.*len(shuffled_dists))
    print("animal {}: fraction {:.4f}".format(a, frac_larger_shuf))
# %% codecell
# polar plot to show angles and distances
f = figure(figsize=(6,2))
gs = plt.GridSpec(1,2, wspace=0.5)
ax = plt.subplot(gs[0],projection='polar')
for a in animals:
    lno = ax.plot([0,animal_results[a]["a_r_b"]],
                  [0,animal_results[a]["d_r_b"]],
                  "r-o",
                  markersize=.6,
                  lw=.6,
                  zorder=10)
lno[0].set_label("observed")
for s in shuffles:
    for a in animals:
        lns = ax.plot([0,s[a]["a_r_b"]],
                      [0,s[a]["d_r_b"]],
                      "b-",
                      linewidth=0.3,
                      zorder=9)
lns[0].set_label('shuffled')
ax.legend(fontsize=6, loc=(.9,0.2))
ax.set_yticks([100,200,300])
ax.text(.25,320.,"distance [$\mu$m]", fontsize=6)
#histograms of shuffled and observed distances
ax = f.add_subplot(gs[1], frame_on=False)
bins = np.linspace(0, np.max(shuffled_dists), 50)
width = bins[1]*.5
shufdisthist, _ = np.histogram(shuffled_dists, bins)
shuf_l = ax.bar(bins[:-1], shufdisthist, width=width, label="shuffled")
ylim = ax.get_ylim()
label='observed'
for a in animals:
    d = animal_results[a]["d_r_b"]
    obs_l = ax.plot([d,d], ylim, "r-", label=label, lw=0.5)
    label=""
ax.legend(fontsize=6)
ax.set_xlabel("distance [$\mu$m]")
ax.set_ylabel("frequency")

# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'patch_centre_directions_and_distances.svg')
f.savefig(savename, bbox_inches='tight', dpi=600)
# %% codecell
#per animal
f = figure(figsize=(4,9))
gs = plt.GridSpec(len(animals), 2, wspace=0.4, hspace=0.4, width_ratios=[.3,.7])
abins = np.arange(0,2*np.pi, np.pi/12.)
dbins = np.arange(0,301,10)
dec= False
for i,a in enumerate(animals):
    a_dist = animal_results[a]["d_r_b"]
    a_angle = animal_results[a]["a_r_b"]

    a_shuffled_dists = np.array([s[a]["d_r_b"] for s in shuffles])
    a_shuffled_angles = np.array([s[a]["a_r_b"] for s in shuffles])

    #fraction of shuffled distances larger than the average observed distance, and per animal
    frac_larger_shuf = np.sum(a_shuffled_dists > a_dist)/(1.*len(a_shuffled_dists))
    print("animal {}: fraction {:.3f}".format(a, frac_larger_shuf))

    angles_hist, _ = np.histogram(a_shuffled_angles, abins)
    dists_hist, _ = np.histogram(a_shuffled_dists, dbins)

    aax = f.add_subplot(gs[i,0], projection='polar', frame_on=False)
    aax.bar(abins[:-1], angles_hist, width=np.pi/12.)
    aax.plot([0,a_angle], [0,max(angles_hist)], "r-", lw=0.6)
    aax.tick_params(axis='both', which='major', labelsize=6)
    aax.set_yticks([])
    aax.set_ylim(0, max(angles_hist))

    dax = f.add_subplot(gs[i,1], frame_on=False)
    dax.bar(dbins[:-1], dists_hist, width=dbins[1]-dbins[0]-1, label='shuffled')
    dax.plot([a_dist, a_dist], [0,max(dists_hist)*.8], 'r-', lw=0.6, label='observed')
    dax.text(a_dist,
             max(dists_hist)*.85,
             "$p$={:00.3g}".format(frac_larger_shuf),
             fontsize=6,
             horizontalalignment='center')
    dax.tick_params(axis='both', which='major', labelsize=6)
    if not dec:
        dax.legend(fontsize=6)
        dax.set_xlabel("distance [$\mu$m]", fontsize=6)
        dax.set_ylabel("frequency", fontsize=6)
        dec = True




# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'patch_centre_directions_and_distances_per_animal.svg')
f.savefig(savename, bbox_inches='tight', dpi=600)
# %% markdown
#
# ### Supplemental Figure
# %% codecell
cluster = json.load(open(cluster_file))[title] if load_cluster else None
# %% codecell
fig_dim = (7.48,9.4)
global_fs= 7

layout = {   'axes.labelsize': 8,
             'axes.linewidth': .75,
             'xtick.major.size': 2,     # major tick size in points
             'xtick.minor.size': 1,     # minor tick size in points
             'xtick.labelsize': 8,       # fontsize of the tick labels
             'xtick.major.pad': 2,

             'ytick.major.size': 2,      # major tick size in points
             'ytick.minor.size': 1,     # minor tick size in points
             'ytick.labelsize':8,       # fontsize of the tick labels
             'ytick.major.pad': 2,

             'mathtext.default' : 'regular',
             'legend.fontsize': 8
             }

import matplotlib as mpl
for k, v in layout.items():
    mpl.rcParams[k] = v
# %% codecell
num_cluster = len(cluster)
num_animals = len(animals)
num_stim = len(ts.label_stimuli)
clust_colors = {clust: plt.cm.gist_rainbow(1.*cluster.index(clust)/num_cluster) for clust in cluster}

fig = plt.figure(figsize=fig_dim)
gs_meta = matplotlib.gridspec.GridSpec(3, 1, bottom=0.1, top = 0.99, left = 0.05, right=0.95, height_ratios=[2,2,0.25*num_cluster], hspace=0.01)

# plot dendrogarm
gs_top = matplotlib.gridspec.GridSpecFromSubplotSpec(1,1, gs_meta[0])
ax = fig.add_subplot(gs_top[0])
lw = matplotlib.rcParams['lines.linewidth']
matplotlib.rcParams['lines.linewidth'] = 0.6
d = dendrogram(link, link_color_func = color_clusters(cluster, clust_colors), count_sort='descending')
matplotlib.rcParams['lines.linewidth'] = lw
ax.set_xticks([])
ax.set_yticks([0,0.3,1])
ax.set_ylabel('corr. distance')
ax.set_xlabel('putative glomeruli')
ax.text(-225,1,'A)', weight='bold')

# plot locations
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, num_animals, gs_meta[1])
axbase = {animal: fig.add_subplot(gs[0,ix]) for ix, animal in enumerate(animals)}
plot_location(ts_normed, link, cluster, bg_dict, axbase, clust_colors, scalebar=False)
axbase[animals[0]].text(-37.5,0,'B)', weight='bold')

# add scalebar
pixel_size = 1.63/1344. *1000. #µm
len_100 = 100.*pixel_size   # 100 µm
for a in animals:
    bg = bg_dict[a]
    bgs = bg.shape
    # extent=[0,84,64,0])
    scalefacx = 84./bgs[0]
    axbase[a].plot((100*scalefacx,(100+len_100)*scalefacx),(55,55),'w')

# prepare axes for plotting spectra
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(num_cluster,2, gs_meta[2], hspace=0, wspace=0.04, width_ratios=[.97, .03])
axtime = OrderedDict([(clust,fig.add_subplot(gs[ix,0])) for ix, clust in enumerate(cluster)])
plot_spec(ts_normed, link, cluster, axtime, clust_colors, mean_plot_heatmap)
axtime[cluster[0]].text(-3.1,0, 'C)', weight='bold')
axtime[cluster[-1]].set_xlabel("odorants")
axtime[cluster[len(cluster)//2]].set_ylabel("putative glomeruli")

# colorbar
cbar = []
for i, clust in enumerate(cluster):

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('tmp', ['0',clust_colors[clust]])
    cbar.append([cmap(i) for i in np.arange(1,0,-0.01)])

axbar = fig.add_subplot(gs[:,1])
axbar.imshow(np.array(cbar).swapaxes(0,1), interpolation='none', aspect='auto')
axbar.set_xticks([])
axbar.set_yticks([-0.5,99])
axbar.set_yticklabels(['max', '0'])
axbar.yaxis.set_ticks_position('right')
axbar.set_ylabel('response strength', labelpad=-9)
axbar.yaxis.set_label_position('right')
# %% codecell
savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', 'Supp_More_tunotopic_neighbors.png')
fig.savefig(savename, bbox_inches='tight', dpi=600)
# %% markdown
# ### Visualization of other cluster combinations of interest
# Just for information, not part of the manuscript
# %% codecell
#cluster = [200, 254, 289,204,279,258,220,183,281,257] #all tunotopic close clusters to MOR18-2
#cluster = [200, 163, 247, 254, 197, 319, 281, 362] #good models

cluster = [200,250,211,172,300,264,257,292,261,131,146,62] #all spatial close clusters to MOR18-2

num_cluster = len(cluster)
num_animals = len(animals)
num_stim = len(ts.label_stimuli)
clust_colors = {clust: plt.cm.gist_rainbow(1.*cluster.index(clust)/num_cluster) for clust in cluster}

fig = plt.figure(figsize=(15,1*(num_cluster+2)))
gs_meta = matplotlib.gridspec.GridSpec(3, 1, bottom=0.1, top = 0.99, left = 0.05, right=0.95, height_ratios=[1,1,num_cluster], hspace=0.05)

gs_top = matplotlib.gridspec.GridSpecFromSubplotSpec(2,1, gs_meta[0], hspace=0.01, height_ratios=[4,1])
# plot dendrogram
ax = fig.add_subplot(gs_top[0])
lw = matplotlib.rcParams['lines.linewidth']
matplotlib.rcParams['lines.linewidth'] = 0.6
d = dendrogram(link, link_color_func = color_clusters(cluster, clust_colors), count_sort='descending')
matplotlib.rcParams['lines.linewidth'] = lw
ax.set_xticks([])
ax.set_yticks([0,0.4,1])

# plot MOR18-2 correlation
ax = fig.add_subplot(gs_top[1])
ax.imshow((1-cor[d['leaves']]).reshape((1,-1)), cmap= plt.cm.RdYlGn, interpolation='none', aspect='auto', vmin=-1, vmax=1)
ax.set_yticks([])

# plot locations
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(1, num_animals, gs_meta[1])
axbase = {animal: fig.add_subplot(gs[0,ix]) for ix, animal in enumerate(animals)}
plot_location(ts_normed, link, cluster, bg_dict, axbase, clust_colors)

# prepare axes for plotting spectra
gs = matplotlib.gridspec.GridSpecFromSubplotSpec(num_cluster,1, gs_meta[2], hspace=0)
axtime = OrderedDict([(clust,fig.add_subplot(gs[ix])) for ix, clust in enumerate(cluster)])
plot_spec(ts_normed, link, cluster, axtime, clust_colors, percentile_plot, norm=np.sqrt(np.sum(ts_normed._series**2, 0)))

#savename = os.path.join(toplevelpath, 'glomcentric_code', 'results', '_other_cluster_combinations.png')
#fig.savefig(savename, bbox_inches='tight', dpi=600)
