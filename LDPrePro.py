import numpy as np
from Cloud import CRCal

def Landsat_Chunks_Extents(Img_Row_Dim,Img_Col_Dim,Chunk_Size):
    Chunks_R_N,Chunks_C_N=int(Img_Row_Dim/Chunk_Size),int(Img_Col_Dim/Chunk_Size)    
        
    Chunks_R_InterVals=np.linspace(0,Chunks_R_N*Chunk_Size,Chunks_R_N+1).astype(int)
    Chunks_C_InterVals=np.linspace(0,Chunks_C_N*Chunk_Size,Chunks_C_N+1).astype(int)
    Chunks_R_InterVals[-1]=Chunks_R_InterVals[-1]+Img_Row_Dim%Chunk_Size
    Chunks_C_InterVals[-1]=Chunks_C_InterVals[-1]+Img_Col_Dim%Chunk_Size
    
    Chunks_R_Extent=np.concatenate((Chunks_R_InterVals[:-1,np.newaxis],Chunks_R_InterVals[1:,np.newaxis]),axis=1)
    Chunks_C_Extent=np.concatenate((Chunks_C_InterVals[:-1,np.newaxis],Chunks_C_InterVals[1:,np.newaxis]),axis=1)
    Chunks_RC_Extent=np.ones((len(Chunks_R_Extent)*len(Chunks_C_Extent),4)).astype(int)

    for i in range(len(Chunks_R_Extent)):
        for j in range(len(Chunks_C_Extent)):
            Chunks_RC_Extent[len(Chunks_C_Extent)*i+j]=np.concatenate((Chunks_R_Extent[i],Chunks_C_Extent[j]))
    return Chunks_RC_Extent

def Landsat_Chunks_PreProcess(Chunks_RC_Extent,Landsat_Imgs_3D,Cloud_ID):
    Chunks_List=[]
    for chunk_i in range(len(Chunks_RC_Extent)):
        Chunk_Tmp_3D=Landsat_Imgs_3D[:,Chunks_RC_Extent[chunk_i,0]:Chunks_RC_Extent[chunk_i,1],
                                       Chunks_RC_Extent[chunk_i,2]:Chunks_RC_Extent[chunk_i,3]]
        Chunk_Tmp_Cloud_Ratio=CRCal(Chunk_Tmp_3D,Cloud_ID)
        Chunk_Tmp_3D[Chunk_Tmp_Cloud_Ratio>10]=Cloud_ID
        Chunks_List.append(Chunk_Tmp_3D)
    return Chunks_List

def Landsat_Imgs_PreProcess(Landsat_Imgs_3D,Chunk_Size,Cloud_ID):
    _,Img_Row_Dim,Img_Col_Dim=Landsat_Imgs_3D.shape
    Chunks_RC_Extent=Landsat_Chunks_Extents(Img_Row_Dim,Img_Col_Dim,Chunk_Size)
    Landsat_Chunks_3D_List=Landsat_Chunks_PreProcess(Chunks_RC_Extent,Landsat_Imgs_3D,Cloud_ID)
    
    for i,chunk_extent in enumerate(Chunks_RC_Extent):
        Landsat_Imgs_3D[:,chunk_extent[0]:chunk_extent[1],chunk_extent[2]:chunk_extent[3]]=Landsat_Chunks_3D_List[i]
    return Landsat_Imgs_3D

