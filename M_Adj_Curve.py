import mkl,time
import numpy as np
import sacpy as scp
import statsmodels.api as sm
from MatrixR import corr2_coeff

mkl.set_num_threads(1)
Cloud_ID=-9 #云标识

def Search_Similar_PixelsID(Corr_1D,R_Max):
    M_Similar_PixelsN_Tmp=len(np.argwhere(Corr_1D>R_Max)[:,0]) # 寻找到的MODIS相似像元阈值判定
    Maxi_N=10
    M_Similar_PixelsID=np.array([])
    if M_Similar_PixelsN_Tmp>=Maxi_N:
        M_Similar_PixelsID=np.argsort(-Corr_1D)[:Maxi_N]
    elif M_Similar_PixelsN_Tmp>=3:
        M_Similar_PixelsID=np.argsort(-Corr_1D)[:M_Similar_PixelsN_Tmp]
    
    return M_Similar_PixelsID

def M_Ref_Adj_TS_Cal(L_M_Tup,Pixel_ID):
    'step 01'
    L_2D,M_2D=L_M_Tup[0],L_M_Tup[1]
    #Win_R_Tmp=int(np.sqrt(L_2D.shape[0])/2)
    
    L_NDVI=L_2D[Pixel_ID]
    L_Rep_N=len(M_2D)
    #L_NDVI=L_2D[Center_PixelID_0]
    L_NDVI_2D=np.repeat(L_NDVI[None],L_Rep_N,axis=0)

    L_NDVI_Valid=L_NDVI[L_NDVI!=Cloud_ID][:,None]
    M_2D_Valid=M_2D[L_NDVI_2D!=Cloud_ID].reshape(L_Rep_N,-1)
    
    #time0=time.time()
    #Corr_1D=np.corrcoef(L_NDVI_Valid,M_2D_Valid)[:L_Rep_N,L_Rep_N:][0]
    Corr_1D=scp.multi_corr(L_NDVI_Valid,M_2D_Valid.T)
    Corr_1D=Corr_1D[:,0,1]
    if len(L_NDVI_Valid)==2:
        Corr_1D[Corr_1D==1]=0
    #Corr_1D=corr2_coeff(L_NDVI_Valid,M_2D_Valid)[0]

    #N=Win_R_Tmp+5
    #Corr_1D=np.array([np.corrcoef(L_NDVI[L_NDVI!=Cloud_ID],m_ndvi[L_NDVI!=Cloud_ID])[0,1] for m_ndvi in M_2D])
    #time1=time.time()
    #M_PixelsID=np.arange(0,len(L_2D))
    #M_Similar_PixelsN_Tmp=len(np.argwhere(Corr_1D>0.9)[:,0]) # 寻找到的MODIS相似像元阈值判定
    
    #print('step01 corr :{t}s'.format(t=time1-time0))

    'step 02'
    M_Similar_PixelsID=Search_Similar_PixelsID(Corr_1D,0.8)
    Flag_Similar=1
    if len(M_Similar_PixelsID)<=3:
        M_Similar_PixelsID=Search_Similar_PixelsID(Corr_1D,0.6)
        if len(M_Similar_PixelsID)<=3:
            M_Similar_PixelsID=np.argwhere(~np.isnan(Corr_1D))[:,0]
            Flag_Similar=0


    '''
    M_Similar_PixelsN_Tmp=len(np.argwhere(Corr_1D>0.8)[:,0]) # 寻找到的MODIS相似像元阈值判定
    Maxi_N=10
    if M_Similar_PixelsN_Tmp>Maxi_N:
        M_Similar_PixelsID=np.argsort(-Corr_1D)[:Maxi_N]
        M_Similar_PixelsID=M_Similar_PixelsID[~np.isnan(M_Similar_PixelsID)]
    
    M_Similar_PixelsID=np.argsort(-Corr_1D)[:10] # Maximum similar number
    M_Similar_PixelsCorr=Corr_1D[M_Similar_PixelsID]
    M_Similar_PixelsCorr=M_Similar_PixelsCorr[~np.isnan(M_Similar_PixelsCorr)]

    M_Similar_PixelsID=M_Similar_PixelsID[:len(M_Similar_PixelsCorr)]
    #print(M_Similar_PixelsID.shape)
    
    
    if M_Similar_PixelsN_Tmp>25:
        M_Similar_PixelsID=np.argsort(-Corr_1D)[:20]
        M_Similar_PixelsID=M_Similar_PixelsID[~np.isnan(M_Similar_PixelsID)]
    elif len(M_Similar_PixelsID)>5:
        pass
    else:
        M_Similar_PixelsID=np.argwhere(Corr_1D>0.6)[:,0]
        if len(M_Similar_PixelsID)<=3:
            M_Similar_PixelsID=M_PixelsID[np.argsort(Corr_1D)[-6:]]
        else:
            pass
    '''

    'step 03'
    M_Similar_PixelsTS=M_2D[M_Similar_PixelsID]
    if Flag_Similar==1:
        Corr_Similar=Corr_1D[M_Similar_PixelsID]
        R_Similar=(Corr_Similar-np.min(Corr_Similar))/(np.max(Corr_Similar)-np.min(Corr_Similar))
        W_Similar=R_Similar/np.sum(R_Similar)

        M_Ref_TS=np.sum(W_Similar[:,None]*M_Similar_PixelsTS,axis=0)
    else:
        M_Ref_TS=np.mean(M_Similar_PixelsTS,axis=0)

    'step 04'
    #time0=time.time()
    X=sm.add_constant(M_Ref_TS[L_NDVI!=Cloud_ID])
    LM=sm.OLS(L_NDVI[L_NDVI!=Cloud_ID],X).fit()
    KB=LM.params
    M_Adj_TS=KB[1]*M_Ref_TS+KB[0]
    #time1=time.time()
    #print('step04:{t}s'.format(t=time1-time0))
    return M_Similar_PixelsID,M_Ref_TS,M_Adj_TS 

def M_Adj_TS_Solve(L_2D,M_2D,L_M_All_Corr_2D,Pixel_ID):
    'step 01'
    '''
    L_2D_Copy=np.copy(L_2D)
    L_2D_Copy=L_2D_Copy.astype(float)
    L_2D_Copy[L_2D_Copy==Cloud_ID]=np.nan
    '''
    
    L_NDVI=L_2D[Pixel_ID]
    L_M_Center_Corr_1D=L_M_All_Corr_2D[Pixel_ID]
    L_M_Center_Similar_PixelsID=Search_Similar_PixelsID(L_M_Center_Corr_1D,0.8)

    'step 02'
    Flag_Similar=1
    if len(L_M_Center_Similar_PixelsID)<=3:
        L_M_Center_Similar_PixelsID=Search_Similar_PixelsID(L_M_Center_Corr_1D,0.6)
        if len(L_M_Center_Similar_PixelsID)<=3:
            L_M_Center_Similar_PixelsID=np.argwhere(~np.isnan(L_M_Center_Corr_1D))[:,0]
            Flag_Similar=0

    'step 03'
    #L_M_Center_Similar_Corr_1D=L_M_Center_Corr_1D[L_M_Center_Similar_PixelsID]
    L_M_Similar_PixelsTS=M_2D[L_M_Center_Similar_PixelsID]
    if Flag_Similar==1:
        Corr_Similar=L_M_Center_Corr_1D[L_M_Center_Similar_PixelsID]
        R_Similar=(Corr_Similar-np.min(Corr_Similar))/(np.max(Corr_Similar)-np.min(Corr_Similar))
        W_Similar=R_Similar/np.sum(R_Similar)

        M_Ref_TS=np.sum(W_Similar[:,None]*L_M_Similar_PixelsTS,axis=0)
    else:
        M_Ref_TS=np.mean(L_M_Similar_PixelsTS,axis=0)

    'step 04'
    #time0=time.time()
    X=sm.add_constant(M_Ref_TS[L_NDVI!=Cloud_ID])
    LM=sm.OLS(L_NDVI[L_NDVI!=Cloud_ID],X).fit()
    KB=LM.params
    M_Adj_TS=KB[1]*M_Ref_TS+KB[0]

    return L_M_Center_Similar_PixelsID,M_Adj_TS

