"""
Reassign large BH with very small galaxy mass into the nearest central subhalo
criterion: M_BH > M_gal/10, M_BH>2e7
- rearrange the BHs in the target group, so that the large BH is in subgroup0
- (order BH in each subgroup by mass maybe later)
- rewrite col5 of SubhaloLenType to be consistent with the new ordering (+1 for central, -1 for small gal)
- rewrite SubhaloOffsetType to be consistent with SubhaloLenType
- should only concern some of the large groups so it should be a small job?
"""

import numpy as np
from bigfile import BigFile,FileMPI
import h5py
import sys,os
import glob
import matplotlib
import matplotlib.pyplot as plt
import argparse

cmap = plt.get_cmap("Set2")
# from bf_util import *
# from mpi4py import MPI

hh = 0.6774

def get_cmbh_mgal(pig2):
    ngal = int(1e7) # this is sufficient
    nbh = int(1e8)
    
    #------------- Load data ------------------
    gal_mass = pig2['SubGroups/SubhaloMassType'][:ngal][:,4] * 1e10/hh
    sublen5 = pig2['SubGroups/SubhaloLenType'][:ngal][:, 5]
    mask = gal_mass > 0
    mask &= sublen5 > 0
    gal_mass = gal_mass[mask]
    sublen5 = sublen5[mask]
    suboff5 = pig2['SubGroups/SubhaloOffsetType'][:ngal][mask, 5]

    ind = mask.nonzero()[0]
    Ninit = len(ind)

    # load BH properties
    bhmass = pig2['5/BlackholeMass'][:nbh] * 1e10/hh
    gal_cm5 = np.array([np.max(bhmass[suboff5[i] : suboff5[i] + sublen5[i]]) for i in range(Ninit)])
    
    bhgroup = pig2['5/GroupID'][:nbh] - 1
    gal_gidx = np.array([bhgroup[suboff5[i]] for i in range(Ninit)])
    
    cen_idx = np.array([suboff5[i] + np.argmax(bhmass[suboff5[i] : suboff5[i] + sublen5[i]]) for i in range(Ninit)])
    return gal_gidx, gal_cm5, gal_mass, cen_idx, ind
    


def get_targets(gal_gidx, gal_cm5, gal_mass, cen_idx, ind):

    # reassign these BHs:
    mask1 = gal_cm5 > gal_mass/30
    mask1 &= gal_cm5 > 5e6
    
    gidx_tar = gal_gidx[mask1]
    sidx_tar = ind[mask1]
    bidx_tar = cen_idx[mask1]
    # print(gidx_tar[:10])
    # print(sidx_tar[:10])
    return gidx_tar, sidx_tar, bidx_tar


def calc_dr(cen_pos, bhpos):
    dpos = cen_pos - bhpos
    lbox = 250000.
    dpos[dpos > lbox/2] -= lbox
    dpos[dpos < -lbox/2] += lbox
    dr = np.linalg.norm(dpos, axis=1)
    return dr





def process_group(pig_r, gidx):
    
    gstart, gend = int(Gobt[gidx][5]), int(Gobt[gidx][5] + Glbt[gidx][5]) # abs start, end index of BH in this group
    firstsub = int(FirstSub[gidx])
    nsub = int(Nsub[gidx])
    print(nsub)
    
    sobt    = Sobt[firstsub : firstsub + nsub] # start index of all part in each subgroup
    slbt    = Slbt[firstsub : firstsub + nsub]
    slen    = pig_r['SubGroups/SubhaloLen'][firstsub : firstsub + nsub]
    cen_pos = Spos[firstsub : firstsub + nsub] # position of each subgroup
    
    # the original index of all bhs in this group
    allbh_sidx = np.zeros(Glbt[gidx][5], dtype=int)
    for nn in range(nsub):
        beg, end = int(sobt[nn][5] - gstart), int(sobt[nn][5] - gstart + slbt[nn][5])
        allbh_sidx[beg : end] = nn
    allbh_sidx[end:] = 100000000
    
    print('BHs to reassign in this group:',  len(groups[gidx]), flush=True)
    
    for sidx, bidx in groups[gidx]:
        bhpos   = pig_r['5/Position'][bidx]
        bhgroup = pig_r['5/GroupID'][bidx] - 1
        bhmass  = pig_r['5/BlackholeMass'][bidx] * 1e10/hh
        #--------------------Find new order----------------------------
        idx_in_group = int(bidx - gstart) # bh index in group
        dr = calc_dr(cen_pos, bhpos)

        for i,d in enumerate(dr):
            if d < 30:
                break
        newsub = i + firstsub # abs index of new subgroup
        
        if Smbt[newsub][4]*1e10/hh < bhmass * 100:
            # relax distance and redo search
            for i,d in enumerate(dr):
                if d < 100:
                    break
            newsub = i + firstsub 
            
        
        print('BH with mass %.1e will be reassigned to subgroup %d with separation %.1f ckpc/h, mass %.1e'\
              %(bhmass, i, d, Smbt[newsub][4]*1e10/hh), flush=True)
        bhmass_list.append(bhmass)
        smass_list.append(Smbt[newsub][4]*1e10/hh)
        
        allbh_sidx[bidx - gstart] = i
        
        slbt[sidx - firstsub][5] -= 1
        slen[sidx - firstsub] -= 1
        slbt[i][5] += 1
        slen[i] += 1

    # this is the new order of BHs in this group
    order = np.argsort(allbh_sidx)

    sobt = np.zeros(slbt.shape, dtype=np.int64)
    sobt[1:] = np.cumsum(slbt[:-1], axis=0)
    sobt += Sobt[firstsub]
    
#     print(gstart, sobt[0])
#     print(allbh_sidx)
#     print(order)
    
    return order, sobt, slbt, slen


def rewrite_group(pig_r, pig_w, gidx):
    order, sobt, slbt, slen = process_group(pig_r, gidx)
    gstart, gend = Gobt[gidx][5], Gobt[gidx][5] + Glbt[gidx][5] # abs start, end index of BH in this group
    firstsub = FirstSub[gidx]
    
    print('Reordering BHs in group...', flush=True)
    for ff in glob.glob(outdir + '/5/*'):
        blockname = ff.split('/')[-1]
        data = pig_r['5/%s'%blockname][gstart:gend]
        data = data[order]
        pig_w['5/%s'%blockname].write(gstart,data)
        
    
    print('Rewriting Length and Offset of BHs...', flush=True)
    
    pig_w['SubGroups/SubhaloLenType'].write(firstsub, slbt)
    pig_w['SubGroups/SubhaloLen'].write(firstsub, slen)
    pig_w['SubGroups/SubhaloOffsetType'].write(firstsub, sobt)
    
    print('done with group %d'%(gidx), flush=True)
    
    # return newsub, bmass, smass[newsub][4]*1e10/hh
    



if __name__ == "__main__":

    #-------------- Cmd line Args ------------------------------
    parser = argparse.ArgumentParser(description='subgrpID-subfind')
    parser.add_argument('--snap',required=True,type=int,help='snapshot number to process')

    args = parser.parse_args()
    snap = int(args.snap)
    indir = '/hildafs/datasets/Asterix/PIG2/PIG_%03d_subfind'%(snap)
    outdir = '/hildafs/datasets/Asterix/PIG2/PIG_%03d_test'%(snap)
    
    pig_r = BigFile(indir)
    pig_w = BigFile(outdir)
    
    
    Gobt = pig_r['FOFGroups/OffsetByType']
    Glbt = pig_r['FOFGroups/LengthByType']
    
    Sobt = pig_r['SubGroups/SubhaloOffsetType']
    Slbt = pig_r['SubGroups/SubhaloLenType']
    
    Spos = pig_r['SubGroups/SubhaloPos']
    Smbt = pig_r['SubGroups/SubhaloMassType']
    
    FirstSub = pig_r['FOFGroups/GroupFirstSub']
    Nsub     = pig_r['FOFGroups/GroupNsubs']

    
    
    gal_gidx, gal_cm5, gal_mass, cen_idx, ind = get_cmbh_mgal(pig_r)
    gidx_tar, sidx_tar, bidx_tar = get_targets(gal_gidx, gal_cm5, gal_mass, cen_idx, ind)
    
    
    # group by gidx
    groups = {}
    for gidx, sidx, bidx in zip(gidx_tar, sidx_tar, bidx_tar):
        if gidx in groups:
            groups[gidx].append((sidx, bidx))
        else:
            groups[gidx] = [(sidx, bidx)]
    
    
    
    print('total BHs to reassign:', len(gidx_tar), 'number of groups:', len(groups), flush=True)
    
#---------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1,2,figsize=(12,5), sharex=True, sharey=True)
    mask1 = gal_cm5 > gal_mass/10
    mask1 &= gal_cm5 > 2e7
    ax[0].scatter(gal_cm5, gal_mass, s=2, alpha=0.2, color=cmap(0))
    ax[0].scatter(gal_cm5[mask1], gal_mass[mask1], s=7, alpha=0.8, color=cmap(1))
    ax[0].set(xscale='log', yscale='log')
    ax[0].set(xlabel=r'$M_{\rm BH}\,[M_\odot]$', ylabel=r'$M_{\rm gal}\,[M_\odot]$')
    ax[1].set(xlabel=r'$M_{\rm BH}\,[M_\odot]$')
#----------------------------------------------------------------------------------------
    
    bhmass_list = []
    smass_list = []
    for gidx in list(groups.keys())[:]:
        rewrite_group(pig_r, pig_w, gidx)

        
        
    # plot again to check
    print('Checking new data...', flush=True)
    gal_gidx, gal_cm5, gal_mass, gal_gidx, cen_idx = get_cmbh_mgal(pig_w)
    
    
    ax[1].scatter(gal_cm5, gal_mass, s=2, alpha=0.2, color=cmap(0))
    # ax[1].scatter(gal_cm5_sel, gal_sel, s=7, alpha=0.8, color=cmap(1))
    ax[1].scatter(bhmass_list, smass_list, s=7, alpha=0.8, color=cmap(1))
    
    
    plt.savefig('/hildafs/home/nianyic/Astrid_analysis/subfind/plot_check/mbh_mgal_snap%03d.png'%(snap))
    
    print('done!', flush=True)
    
    
    
    
    
    
    
