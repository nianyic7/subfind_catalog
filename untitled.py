outdir = '/hildafs/datasets/Asterix/PIG2/PIG_214_subfind'
pig2 = BigFile(outdir)

gLength = pig2['FOFGroups/LengthByType']
gOffset = pig2['FOFGroups/OffsetByType']
FirstSub = pig2['FOFGroups/GroupFirstSub']

sLength = pig2['SubGroups/SubhaloLenType']
sOffset = pig2['SubGroups/SubhaloOffsetType']
sMass = pig2['SubGroups/SubhaloMassType']

NSubs = pig2['FOFGroups/GroupNsubs']



BHMass = pig2['5/BlackholeMass'][:]*1e10/hh
bidxlist, = np.where(BHMass > 1e6)

bmass = BHMass[bidxlist]
del BHMass
print(len(bidxlist))

suboff = pig2['SubGroups/SubhaloOffsetType'][:]
suboff5 = suboff[:,5]
del suboff

def place(bidxlist,suboff5):
    sidxlist = np.searchsorted(suboff5,bidx,side='right')-1
    return sidxlist

def get_offset(bidxlist,centers):
    dpos = pig2['5/Position'][:][bidxlist] - centers
    dpos[dpos > box/2] -= box
    dpos[dpos < -box/2] += box
    return np.linalg.norm(dpos,axis=1)

box = 250000.
# this line get you the subgroup idx of the blackhole
sidxlist = place(bidxlist,suboff5)

# this is the subhalo centers
centers = pig2['SubGroups/SubhaloPos'][:][sidxlist]
roff = get_offset(bidxlist,sidxlist)

plt.scatter(bmass,roff/4,s=0.1)
plt.xscale('log')
plt.xlabel(r'$M_{\rm BH}\,[M_\odot]$')
plt.ylabel(r'$r_{\rm off}\,[kpc]$')
plt.yscale('log')