import numpy as np
from bigfile import BigFile,FileMPI
import h5py
import sys,os
import glob
import argparse
from bf_util import *
from mpi4py import MPI
from scipy.stats import rankdata

"""
assign subgroupID to BHs
also rewrite mbt cols using True BH mass


"""



def place(bidxlist,suboff5):
    sidxlist = np.searchsorted(suboff5,bidx,side='right')-1
    return sidxlist



#--------------------------------------------------------------------------
 
if __name__ == "__main__":
    #----------- MPI Init ----------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()


    #-------------- Cmd line Args ------------------------------
    parser = argparse.ArgumentParser(description='subgrpID-subfind')
    parser.add_argument('--dest',required=True,type=str,help='path of the output file directory')
    parser.add_argument('--gstart',default=0,type=int,help='where to begin the rewriting')
    parser.add_argument('--gend',default=0,type=int,help='where to finish the rewriting')
    

    
    args = parser.parse_args()
    
    #--------------------------
    dest_w  = FileMPI(comm, args.dest, create=True)
    dest_r  = BigFile(args.dest)
    gstart  = int(args.gstart)
    gend    = int(args.gend)

    
    comm.barrier()
    #---------- Initialize  blocks --------------
    blockname = '5/SubgroupIdx'
    dtype = 'i8'
    dsize = dest_r['5/ID'].size
    nfile = dest_r['5/ID'].Nfile
    if gstart == 0:
        block = dest_w.create(blockname, dtype, dsize, nfile)
    else:
        block = dest_w[blockname]
    comm.barrier()
        

    sLength  = dest_r['SubGroups/LengthByType']
    sOffset  = dest_r['SubGroups/OffsetByType']
    # ----------- Split tasks --------------------
    NBHs   = Offset[gend,5]
    istart = NBHs * rank // size
    iend = NBHs * (rank + 1) // size

    print('Rank %d starts from Group %d to Group%d'%(rank,istart,iend),flush=True)
    if rank == 0:
        print('Gstart: %d Gend: %d Total groups:'%(gstart,gend),Ngroups)
        print('Saving dir:',args.dest,flush=True)
    comm.barrier()

    BHidx = np.arange(istart,iend)
    #--------------------------------------------------------------
    p = 5
    
    gidxlist  = dest_r['5/GroupID'][istart:iend]-1
    gstart,gend = min(gidxlist),max(gidxlist)+1
    FirstSub = dest_r['FOFGroups/GroupFirstSub'][gstart:gend]
    
    sstart, send = FirstSub[0], FirstSub[-1] + Nsubs[gend]
    
    suboff = pig2['SubGroups/SubhaloOffsetType'][sstart:send]
    suboff5 = suboff[:,5]
    del suboff

    data = place(bidxlist,suboff5)
    data -= FirstSub[gidxlist-gidxlist[0]]
    print('rank %d, datal length: %d, starting point %d'%(rank,len(data),istart))
    block.write(istart,data)
            
    print('rank %d done!'%rank,flush=True)
    #---------------- copy over the smaller groups ----------------

#     if rank == size//2:
#         print('rank %d copying over the small groups'%rank)
#         print('rstart:',rstart,'dsize:',dsize)
#         rstart = Offset[gend][p]
#         data = - np.ones(dsize-rstart,dtype='i4')
#         block.write(rstart,data)
                



