import numpy as np
from Cloud import CRCal

'Patch Divid: single target pixel'
'Patch Size:1'

def PatchesExtents(Img_Row_Dim,Img_Col_Dim,Patch_Size,Win_R):
    #global Patches_R_N,Patches_C_N
    '''
    Patches_RC_Extent01: normal divide,row and col extent
    Patches_RC_Extent02: extend extent with Radius
    '''
    Patches_R_N,Patches_C_N=int(Img_Row_Dim/Patch_Size),int(Img_Col_Dim/Patch_Size)    
        
    Patches_R_InterVals=np.linspace(0,Patches_R_N*Patch_Size,Patches_R_N+1).astype(int)
    Patches_C_InterVals=np.linspace(0,Patches_C_N*Patch_Size,Patches_C_N+1).astype(int)
    Patches_R_InterVals[-1]=Patches_R_InterVals[-1]+Img_Row_Dim%Patch_Size
    Patches_C_InterVals[-1]=Patches_C_InterVals[-1]+Img_Col_Dim%Patch_Size
    
    Patches_R_Extent=np.concatenate((Patches_R_InterVals[:-1,np.newaxis],Patches_R_InterVals[1:,np.newaxis]),axis=1)
    Patches_C_Extent=np.concatenate((Patches_C_InterVals[:-1,np.newaxis],Patches_C_InterVals[1:,np.newaxis]),axis=1)
    Patches_RC_Extent01=np.ones((len(Patches_R_Extent)*len(Patches_C_Extent),4)).astype(int)
    Patches_RC_Extent02=np.ones((len(Patches_R_Extent)*len(Patches_C_Extent),4)).astype(int)
    for i in range(len(Patches_R_Extent)):
        for j in range(len(Patches_C_Extent)):
            Patches_RC_Extent01[len(Patches_C_Extent)*i+j]=np.concatenate((Patches_R_Extent[i],Patches_C_Extent[j]))
        
    Patches_RC_Extent02[:,0],Patches_RC_Extent02[:,2]=Patches_RC_Extent01[:,0],Patches_RC_Extent01[:,2]
    Patches_RC_Extent02[:,1]=Patches_RC_Extent01[:,1]+Win_R*2
    Patches_RC_Extent02[:,3]=Patches_RC_Extent01[:,3]+Win_R*2
    
    return Patches_RC_Extent01,Patches_RC_Extent02

def PatchesPad(Patches_RC_Extent,Img_4D,Patch_Size,Win_R):
    Img_Row_Dim,Img_Col_Dim=Img_4D.shape[2],Img_4D.shape[3]
    Patches_R_N,Patches_C_N=int(Img_Row_Dim/Patch_Size),int(Img_Col_Dim/Patch_Size)
    Img_4D_Pad=np.pad(Img_4D,((0,0),(0,0),(Win_R,Win_R),(Win_R,Win_R)),'constant',constant_values=-1)
 
    Patches_Pad_List=[]
    for block_i in range(Patches_R_N*Patches_C_N):
        Patches_Pad_List.append(Img_4D_Pad[:,:,
                                          Patches_RC_Extent[block_i,0]:Patches_RC_Extent[block_i,1],
                                          Patches_RC_Extent[block_i,2]:Patches_RC_Extent[block_i,3]])
    return Patches_Pad_List


 

