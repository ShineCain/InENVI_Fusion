import array
import numpy as np
from modape.whittaker import ws2d,ws2doptv

def WWHD_Smooth(Pixel_NDVI_QA_Tup,Iter_Num=1):
    Pixel_NDVI,Pixel_QA=Pixel_NDVI_QA_Tup[0],Pixel_NDVI_QA_Tup[1]
    ## step 01
    Pixel_Initial_Weights=np.ones(len(Pixel_NDVI))
    Pixel_Initial_Weights[Pixel_QA==0]=1.0
    Pixel_Initial_Weights[Pixel_QA==1]=0.3 # 含有噪音，质量低,指示那些预测超过阈值的点
    Pixel_Initial_Weights[Pixel_QA==2]=0.7 # 经过填补后，质量得以提高，要比qa==1高，但低于qa==0
    
    ## step 02
    Good_Points_Percen=np.sum(Pixel_QA==0)/len(Pixel_NDVI)*100
    Marginal_Points_Percen=np.sum(Pixel_QA==2)/len(Pixel_NDVI)*100 # 
    if Good_Points_Percen>=40:
        Weight_Critical=1
    elif (Good_Points_Percen+Marginal_Points_Percen)>=40:
        Weight_Critical=0.5
    else:
        Weight_Critical=0.2
    
    ## step 03
    Y_C=Pixel_NDVI[Pixel_Initial_Weights>=Weight_Critical]
    Y_LU=[np.quantile(Y_C,0.01),np.max(Y_C)]
    Y_GS_SE=0.2*(Y_LU[1]-Y_LU[0])+Y_LU[0]

    ## step 04
    L_Range=array.array('d',np.linspace(-1,1,11))
    L_Optimal=ws2doptv(Pixel_NDVI,Pixel_Initial_Weights,L_Range)[1]
    Pixel_Tmp_Weights=Pixel_Initial_Weights.copy()
    Y_I_Tmp=Pixel_NDVI.copy()
    for i in range(Iter_Num):
        Y_Tmp=np.array(ws2d(Y_I_Tmp,L_Optimal,Pixel_Tmp_Weights))
        R_Tmp=Y_Tmp-Y_I_Tmp
        S_Tmp=np.median(abs(R_Tmp))
        Pixel_Tmp_Weights[np.logical_and(R_Tmp>=0,R_Tmp<6*S_Tmp)]=Pixel_Tmp_Weights[np.logical_and(R_Tmp>=0,R_Tmp<6*S_Tmp)]*(1-(R_Tmp[np.logical_and(R_Tmp>=0,R_Tmp<6*S_Tmp)]/(6*S_Tmp))**2)**2
        Pixel_Tmp_Weights[R_Tmp>=6*S_Tmp]=0.15
        Pixel_Tmp_Weights[Pixel_Tmp_Weights<0.15]=0.15
    
        Y_I_Tmp[np.logical_and(R_Tmp>0,Pixel_NDVI>=Y_GS_SE)]=Y_Tmp[np.logical_and(R_Tmp>0,Pixel_NDVI>=Y_GS_SE)]
    ## step 05
    z=np.array(ws2d(Y_I_Tmp,L_Optimal,Pixel_Tmp_Weights))    
    Pixel_Smmoth_Res=z 
    return Pixel_Smmoth_Res


