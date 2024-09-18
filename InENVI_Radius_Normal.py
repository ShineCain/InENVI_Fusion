import datetime,gc,glob,mkl,natsort,os,sys,time,warnings
import numpy as np
import statsmodels.api as sm
#import multiprocessing as mp
#import matplotlib.pyplot as plt

from LAC import LACF
from osgeo import gdal
from WWHD import WWHD_Smooth
from Model import ModelSelect
from functools import partial
from schwimmbad import MPIPool
from scipy.signal import savgol_filter
from M_Adj_Curve import M_Ref_Adj_TS_Cal
from Divide import PatchesExtents,PatchesPad
from LDPrePro import Landsat_Imgs_PreProcess

Sample_Name,Selected_Model=sys.argv[1],'LGB'
Cloud_ID,Win_R=-9,int(sys.argv[2])
Chunk_Size,Patch_Size=30,1 # Patch_Size---设置为1，中心像元
Start_Year,End_Year=2001,2020

Scale_Factor,Batch_Slices=1000,46
Center_PixelID_0=(2*Win_R+1)**2//2
Win_TargetPixel_Loc=[Win_R,Win_R]
Patch_Pad_Distance_1D=np.array([np.sqrt((i-Win_TargetPixel_Loc[0])**2+(j-Win_TargetPixel_Loc[1])**2) for i in range(2*Win_R+1) for j in range(2*Win_R+1)])
Patch_Pad_Distance_1D[Patch_Pad_Distance_1D==0]=0.1

Select_Start_Year,Select_End_Year=int(sys.argv[3]),int(sys.argv[4])
Years_Num=Select_End_Year-Select_Start_Year+1

if Win_R==0:
    R0_Flag=1
else:
    R0_Flag=0

def InENVI(PatchPadList):
    'step 01'
    #time0=time.time()
    
    Patch_TA_Pad_4D,Patch_SR_Pad_4D=PatchPadList[1],PatchPadList[2]
    Patch_VPD_Pad_4D,Patch_M_NDVI_Pad_4D=PatchPadList[3],PatchPadList[4]
    Patch_ID,Patch_L_NDVI_Pad_4D=PatchPadList[0],PatchPadList[5]

    'step 02'
    _,Time_Len,Patch_Pad_Row,Patch_Pad_Col=Patch_L_NDVI_Pad_4D.shape

    #Patch_M_NDVI_Pad_Pixels_3D=Patch_M_NDVI_Pad_4D.T.reshape(-1,Time_Len,1) # Pixels_Num,Time_Len,1
    Patch_L_NDVI_Pad_Pixels_3D=Patch_L_NDVI_Pad_4D.T.reshape(-1,Time_Len,1)
    Patch_TA_Pad_Pixels_3D=Patch_TA_Pad_4D.T.reshape(-1,Time_Len,1)  
    Patch_SR_Pad_Pixels_3D=Patch_SR_Pad_4D.T.reshape(-1,Time_Len,1)
    Patch_VPD_Pad_Pixels_3D=Patch_VPD_Pad_4D.T.reshape(-1,Time_Len,1)

    'step 03'
    Patch_M_NDVI_Pad_Pixels_2D=Patch_M_NDVI_Pad_4D.squeeze().T.reshape(-1,Time_Len)
    Patch_L_NDVI_Pad_Pixels_2D=Patch_L_NDVI_Pad_4D.squeeze().T.reshape(-1,Time_Len)
    Patch_L_NDVI_Pad_Target_1D=Patch_L_NDVI_Pad_Pixels_2D[Center_PixelID_0]

    #time1=time.time()
    #print('read time:{t}'.format(time1-time0))

    #time2=time.time()
    M_Ref_TS_Cal_Partial=partial(M_Ref_Adj_TS_Cal,(Patch_L_NDVI_Pad_Pixels_2D,Patch_M_NDVI_Pad_Pixels_2D))
    M_Similar_Center_PixelsID,M_Center_Ref_TS,M_Center_Adj_TS=M_Ref_TS_Cal_Partial(Center_PixelID_0)
    #print(M_Similar_Center_PixelsID.shape)
    M_Similar_Center_Adj_TS=np.array([M_Ref_TS_Cal_Partial(similar_id)[2] for similar_id in M_Similar_Center_PixelsID]) # adjusted MODIS-NDVI curve for each pixel
   
    #Patch_ID

    'step 04'
    Patch_L_NDVI_Similar_3D=Patch_L_NDVI_Pad_Pixels_3D[M_Similar_Center_PixelsID] 
    Patch_TA_Similar_3D=Patch_TA_Pad_Pixels_3D[M_Similar_Center_PixelsID]
    Patch_SR_Similar_3D=Patch_SR_Pad_Pixels_3D[M_Similar_Center_PixelsID]
    Patch_VPD_Similar_3D=Patch_VPD_Pad_Pixels_3D[M_Similar_Center_PixelsID]

    Patch_L_NDVI_Center_2D=Patch_L_NDVI_Pad_Pixels_3D[Center_PixelID_0] 
    Patch_TA_Center_2D=Patch_TA_Pad_Pixels_3D[Center_PixelID_0]
    Patch_SR_Center_2D=Patch_SR_Pad_Pixels_3D[Center_PixelID_0]
    Patch_VPD_Center_2D=Patch_VPD_Pad_Pixels_3D[Center_PixelID_0]

    'step 05'
    Patch_L_NDVI_Pad_Annual_DOY=Patch_L_NDVI_Pad_4D.squeeze().reshape(-1,Batch_Slices,Patch_Pad_Row,Patch_Pad_Col).astype(np.float16)
    Patch_L_NDVI_Pad_Annual_DOY[Patch_L_NDVI_Pad_Annual_DOY==Cloud_ID]=np.nan 
    Patch_L_NDVI_Pad_Annual_Ave_3D=np.nanmean(Patch_L_NDVI_Pad_Annual_DOY,axis=0) # 46,5,5

    Patch_L_NDVI_Pad_Annual_Ave_2D=Patch_L_NDVI_Pad_Annual_Ave_3D.T.reshape(-1,Batch_Slices)
    Patch_L_NDVI_Pad_Annual_Ave_2D=np.array(list(map(LACF,list(Patch_L_NDVI_Pad_Annual_Ave_2D))))
    Patch_L_NDVI_Pad_Annual_Ave_TS_3D=np.repeat(Patch_L_NDVI_Pad_Annual_Ave_2D[:,:,None],Years_Num,axis=2).T.reshape(-1,Patch_Pad_Row*Patch_Pad_Col).T[:,:,None]
    Patch_Pad_Annual_Ave_Period_TS_3D=np.repeat(np.repeat(np.arange(1,Batch_Slices+1)[None,],Years_Num,axis=0).ravel()[None,],Patch_Pad_Row*Patch_Pad_Col,axis=0)[:,:,None]

    Patch_PeriodID_Center_2D=Patch_Pad_Annual_Ave_Period_TS_3D[Center_PixelID_0]
    Patch_L_NDVI_Annual_Center_2D=Patch_L_NDVI_Pad_Annual_Ave_TS_3D[Center_PixelID_0]
    Patch_PeriodID_Similar_3D=Patch_Pad_Annual_Ave_Period_TS_3D[M_Similar_Center_PixelsID]
    Patch_L_NDVI_Annual_Similar_3D=Patch_L_NDVI_Pad_Annual_Ave_TS_3D[M_Similar_Center_PixelsID]

    'step 06'
    Center_Concate_2D=np.concatenate((Patch_PeriodID_Center_2D,Patch_TA_Center_2D,Patch_SR_Center_2D,Patch_VPD_Center_2D,M_Center_Adj_TS[:,None],Patch_L_NDVI_Annual_Center_2D,Patch_L_NDVI_Center_2D),axis=1) 
    Similar_Concate_3D=np.concatenate((Patch_PeriodID_Similar_3D,Patch_TA_Similar_3D,Patch_SR_Similar_3D,Patch_VPD_Similar_3D,M_Similar_Center_Adj_TS[:,:,None],Patch_L_NDVI_Annual_Similar_3D,Patch_L_NDVI_Similar_3D),axis=2)
    Similar_Concate_2D=np.vstack([similar_concate for similar_concate in Similar_Concate_3D])

    Center_Train_2D=Center_Concate_2D[Patch_L_NDVI_Center_2D.squeeze()!=Cloud_ID]
    #Center_Test_2D=Center_Concate_2D[Patch_L_NDVI_Center_2D.squeeze()==Cloud_ID]

    Similar_Train_2D=Similar_Concate_2D[Similar_Concate_2D[:,-1]!=Cloud_ID]
    Total_Train_2D=np.concatenate((Similar_Train_2D,Center_Train_2D),axis=0)

    'step 07'
    ML_Model=ModelSelect(Selected_Model)
    Center_Concate_2D[:,-3:]=Center_Concate_2D[:,-3:]/Scale_Factor
    Total_Train_2D[:,-3:]=Total_Train_2D[:,-3:]/Scale_Factor
    ML_Model.fit(Total_Train_2D[:,:-1],Total_Train_2D[:,-1])
    L_Center_Pure_PreRes=ML_Model.predict(Center_Concate_2D[:,:-1])

    'step 08'
    Patch_L_NDVI_Mixed=np.copy(Patch_L_NDVI_Pad_Target_1D)
    Patch_L_NDVI_Mixed=Patch_L_NDVI_Mixed.astype(float)
    Patch_L_NDVI_Mixed[Patch_L_NDVI_Mixed==Cloud_ID]=np.nan
    Patch_L_NDVI_Mixed=Patch_L_NDVI_Mixed/Scale_Factor
    Valid_Min,Valid_Max=np.nanmin(Patch_L_NDVI_Mixed),np.nanmax(Patch_L_NDVI_Mixed)

    L_Center_Pure_PreRes[L_Center_Pure_PreRes<Valid_Min]=Valid_Min
    L_Center_Pure_PreRes[L_Center_Pure_PreRes>Valid_Max]=Valid_Max
    if Patch_ID>=9999 and (Patch_ID+1)%10000==0:
        Batch_Num=int((Patch_ID+1)/10000)
        Batch_Time=datetime.datetime.now()
        print("the finished time of the {batch_num} batch(10000) prediction is {batch_time}".format(batch_num=Batch_Num,batch_time=Batch_Time))    
    
    return L_Center_Pure_PreRes

def InENVI_Normal_V2(PatchPadList):
    'step 01'
    #time0=time.time()
    
    Patch_TA_Pad_4D,Patch_SR_Pad_4D=PatchPadList[1],PatchPadList[2]
    Patch_VPD_Pad_4D,Patch_M_NDVI_Pad_4D=PatchPadList[3],PatchPadList[4]
    Patch_ID,Patch_L_NDVI_Pad_4D=PatchPadList[0],PatchPadList[5]

    'step 02'
    _,Time_Len,Patch_Pad_Row,Patch_Pad_Col=Patch_L_NDVI_Pad_4D.shape

    Patch_M_NDVI_Pad_Pixels_3D=Patch_M_NDVI_Pad_4D.T.reshape(-1,Time_Len,1) # Pixels_Num,Time_Len,1
    Patch_L_NDVI_Pad_Pixels_3D=Patch_L_NDVI_Pad_4D.T.reshape(-1,Time_Len,1)
    Patch_TA_Pad_Pixels_3D=Patch_TA_Pad_4D.T.reshape(-1,Time_Len,1)  
    Patch_SR_Pad_Pixels_3D=Patch_SR_Pad_4D.T.reshape(-1,Time_Len,1)
    Patch_VPD_Pad_Pixels_3D=Patch_VPD_Pad_4D.T.reshape(-1,Time_Len,1)
    
    'step 03'
    Patch_L_NDVI_Pad_3D=Patch_L_NDVI_Pad_4D.squeeze() # time,row,col
    Patch_L_Flag_Pad_3D=np.zeros_like(Patch_L_NDVI_Pad_3D)
    Patch_L_Flag_Pad_3D[Patch_L_NDVI_Pad_3D<Cloud_ID]=1 # true flag--0; water flag--1; cloud flag--2;
    Patch_L_Flag_Pad_3D[Patch_L_NDVI_Pad_3D==Cloud_ID]=2
    Patch_L_Flag_Pad_2D=Patch_L_Flag_Pad_3D.squeeze().T.reshape(-1,Time_Len)

    'step 04'
    Patch_M_NDVI_Pad_Pixels_2D=Patch_M_NDVI_Pad_4D.squeeze().T.reshape(-1,Time_Len)
    Patch_L_NDVI_Pad_Pixels_2D=Patch_L_NDVI_Pad_4D.squeeze().T.reshape(-1,Time_Len)

    'step 05'
    Patch_L_NDVI_Pad_Target_1D=Patch_L_NDVI_Pad_Pixels_2D[Center_PixelID_0]
    Patch_M_NDVI_Pad_Target_1D=Patch_M_NDVI_Pad_Pixels_2D[Center_PixelID_0]
    Patch_L_Flag_Pad_Target_1D=Patch_L_Flag_Pad_2D[Center_PixelID_0]
    
    Target_Flag_01_Ratio=np.sum(Patch_L_Flag_Pad_Target_1D==1)/Time_Len*100
    #print(Target_Flag_01_Ratio)
    if Target_Flag_01_Ratio>90:
        Patch_L_NDVI_Pad_Target_2D=Patch_L_NDVI_Pad_Target_1D.reshape(-1,Batch_Slices).astype(float)
        Patch_L_NDVI_Pad_Target_2D[Patch_L_NDVI_Pad_Target_2D==Cloud_ID]=np.nan
        Patch_L_NDVI_Smooth=Patch_L_NDVI_Pad_Target_1D
        
        if R0_Flag==0:
            M_Ref_TS_Cal_Partial=partial(M_Ref_Adj_TS_Cal,(Patch_L_NDVI_Pad_Pixels_2D,Patch_M_NDVI_Pad_Pixels_2D))
            _,_,M_Center_Adj_TS=M_Ref_TS_Cal_Partial(Center_PixelID_0)
            Patch_L_NDVI_Smooth[Patch_L_Flag_Pad_Target_1D==2]=M_Center_Adj_TS[Patch_L_Flag_Pad_Target_1D==2]
        else:
            Patch_L_NDVI_Smooth[Patch_L_Flag_Pad_Target_1D==2]=Patch_M_NDVI_Pad_Target_1D[Patch_L_Flag_Pad_Target_1D==2]
        
        return Patch_L_NDVI_Smooth

    'step 06'
    Patch_M_NDVI_Center_2D=Patch_M_NDVI_Pad_Pixels_3D[Center_PixelID_0]
    Patch_L_NDVI_Center_2D=Patch_L_NDVI_Pad_Pixels_3D[Center_PixelID_0] 
    Patch_TA_Center_2D=Patch_TA_Pad_Pixels_3D[Center_PixelID_0]
    Patch_SR_Center_2D=Patch_SR_Pad_Pixels_3D[Center_PixelID_0]
    Patch_VPD_Center_2D=Patch_VPD_Pad_Pixels_3D[Center_PixelID_0]

    'step 07'
    Patch_L_NDVI_Pad_Annual_DOY=Patch_L_NDVI_Pad_4D.squeeze().reshape(-1,Batch_Slices,Patch_Pad_Row,Patch_Pad_Col).astype(np.float16)
    Patch_L_NDVI_Pad_Annual_DOY[Patch_L_NDVI_Pad_Annual_DOY==Cloud_ID]=np.nan 
    Patch_L_NDVI_Pad_Annual_Ave_3D=np.nanmean(Patch_L_NDVI_Pad_Annual_DOY,axis=0) # 46,5,5

    Patch_L_NDVI_Pad_Annual_Ave_2D=Patch_L_NDVI_Pad_Annual_Ave_3D.T.reshape(-1,Batch_Slices)
    Patch_L_NDVI_Pad_Annual_Ave_2D=np.array(list(map(LACF,list(Patch_L_NDVI_Pad_Annual_Ave_2D))))
    Patch_L_NDVI_Pad_Annual_Ave_TS_3D=np.repeat(Patch_L_NDVI_Pad_Annual_Ave_2D[:,:,None],Years_Num,axis=2).T.reshape(-1,Patch_Pad_Row*Patch_Pad_Col).T[:,:,None]
    Patch_Pad_Annual_Ave_Period_TS_3D=np.repeat(np.repeat(np.arange(1,Batch_Slices+1)[None,],Years_Num,axis=0).ravel()[None,],Patch_Pad_Row*Patch_Pad_Col,axis=0)[:,:,None]

    Patch_L_NDVI_Pad_Annual_Ave_Smooth_3D=savgol_filter(Patch_L_NDVI_Pad_Annual_Ave_TS_3D[:,:,0],11,3,axis=1)[:,:,None]
    Patch_PeriodID_Center_2D=Patch_Pad_Annual_Ave_Period_TS_3D[Center_PixelID_0]
    Patch_L_NDVI_Annual_Center_2D=Patch_L_NDVI_Pad_Annual_Ave_Smooth_3D[Center_PixelID_0]

    'step 08'
    Patch_L_NDVI_Mixed=np.copy(Patch_L_NDVI_Pad_Target_1D)
    Patch_L_NDVI_Mixed=Patch_L_NDVI_Mixed.astype(float)
    Patch_L_NDVI_Mixed[Patch_L_NDVI_Mixed==Cloud_ID]=np.nan
    Patch_L_NDVI_Mixed=Patch_L_NDVI_Mixed/Scale_Factor
    Valid_Min,Valid_Max=np.nanmin(Patch_L_NDVI_Mixed),np.nanmax(Patch_L_NDVI_Mixed)

    ML_Model=ModelSelect(Selected_Model)
    L_Center_QA=Patch_L_Flag_Pad_Target_1D

    if R0_Flag==1:
        Center_Concate_2D=np.concatenate((Patch_PeriodID_Center_2D,Patch_TA_Center_2D,Patch_SR_Center_2D,Patch_VPD_Center_2D,Patch_M_NDVI_Center_2D,Patch_L_NDVI_Annual_Center_2D,Patch_L_NDVI_Center_2D),axis=1) 
        Center_Concate_2D[:,-3:]=Center_Concate_2D[:,-3:]/Scale_Factor
        
        ML_Model.fit(Center_Concate_2D[:,:-1],Center_Concate_2D[:,-1])

        L_Center_Pure_PreRes=ML_Model.predict(Center_Concate_2D[:,:-1])
        L_Center_Pure_PreRes[L_Center_Pure_PreRes<Valid_Min]=Valid_Min
        L_Center_Pure_PreRes[L_Center_Pure_PreRes>Valid_Max]=Valid_Max
        Patch_L_NDVI_Mixed[np.isnan(Patch_L_NDVI_Mixed)]=L_Center_Pure_PreRes[np.isnan(Patch_L_NDVI_Mixed)]

        Patch_L_NDVI_Smooth=WWHD_Smooth((Patch_L_NDVI_Mixed,L_Center_QA))
        return Patch_L_NDVI_Smooth

    M_Ref_TS_Cal_Partial=partial(M_Ref_Adj_TS_Cal,(Patch_L_NDVI_Pad_Pixels_2D,Patch_M_NDVI_Pad_Pixels_2D))
    M_Similar_Center_PixelsID,M_Center_Ref_TS,M_Center_Adj_TS=M_Ref_TS_Cal_Partial(Center_PixelID_0)

    'step 09'
    #time4=time.time()
    Patch_L_NDVI_Similar_3D=Patch_L_NDVI_Pad_Pixels_3D[M_Similar_Center_PixelsID] 
    Patch_TA_Similar_3D=Patch_TA_Pad_Pixels_3D[M_Similar_Center_PixelsID]
    Patch_SR_Similar_3D=Patch_SR_Pad_Pixels_3D[M_Similar_Center_PixelsID]
    Patch_VPD_Similar_3D=Patch_VPD_Pad_Pixels_3D[M_Similar_Center_PixelsID]
    #time5=time.time()
    #print('step 04 time:{t}'.format(t=time5-time4))

    'step 10'
    Center_Concate_2D=np.concatenate((Patch_PeriodID_Center_2D,Patch_TA_Center_2D,Patch_SR_Center_2D,Patch_VPD_Center_2D,M_Center_Adj_TS[:,None],Patch_L_NDVI_Annual_Center_2D,Patch_L_NDVI_Center_2D),axis=1) 
    Center_Train_2D=np.copy(Center_Concate_2D[Patch_L_NDVI_Center_2D.squeeze()!=Cloud_ID])
    M_Similar_Center_Adj_TS=np.array([M_Ref_TS_Cal_Partial(similar_id)[2] for similar_id in M_Similar_Center_PixelsID]) # adjusted MODIS-NDVI curve for each pixel
    Patch_PeriodID_Similar_3D=Patch_Pad_Annual_Ave_Period_TS_3D[M_Similar_Center_PixelsID]

    Patch_L_NDVI_Annual_Similar_3D=Patch_L_NDVI_Pad_Annual_Ave_Smooth_3D[M_Similar_Center_PixelsID]
    Similar_Concate_3D=np.concatenate((Patch_PeriodID_Similar_3D,Patch_TA_Similar_3D,Patch_SR_Similar_3D,Patch_VPD_Similar_3D,M_Similar_Center_Adj_TS[:,:,None],Patch_L_NDVI_Annual_Similar_3D,Patch_L_NDVI_Similar_3D),axis=2)
    Similar_Concate_2D=np.vstack([similar_concate for similar_concate in Similar_Concate_3D])

    '''
    if R0Flag==1:
        ML_Model.fit(Center_Concate_2D[:,:-1],Center_Concate_2D[:,-1])

        L_Center_Pure_PreRes=ML_Model.predict(Center_Concate_2D[:,:-1])
        L_Center_Pure_PreRes[L_Center_Pure_PreRes<Valid_Min]=Valid_Min
        L_Center_Pure_PreRes[L_Center_Pure_PreRes>Valid_Max]=Valid_Max
        Patch_L_NDVI_Mixed[np.isnan(Patch_L_NDVI_Mixed)]=L_Center_Pure_PreRes[np.isnan(Patch_L_NDVI_Mixed)]

        Patch_L_NDVI_Smooth=WWHD_Smooth((Patch_L_NDVI_Mixed,L_Center_QA))
        return Patch_L_NDVI_Smooth
  
    Patch_L_NDVI_Annual_Center_2D=Patch_L_NDVI_Pad_Annual_Ave_TS_3D[Center_PixelID_0]
    Patch_L_NDVI_Annual_Similar_3D=Patch_L_NDVI_Pad_Annual_Ave_TS_3D[M_Similar_Center_PixelsID]
    '''
    
    #Center_Train_2D=Center_Concate_2D[Patch_L_NDVI_Center_2D.squeeze()!=Cloud_ID]
    #Center_Test_2D=Center_Concate_2D[Patch_L_NDVI_Center_2D.squeeze()==Cloud_ID]

    Similar_Train_2D=Similar_Concate_2D[Similar_Concate_2D[:,-1]!=Cloud_ID]
    Total_Train_2D=np.concatenate((Similar_Train_2D,Center_Train_2D),axis=0)

    'step 11'
    Total_Train_2D[:,-3:]=Total_Train_2D[:,-3:]/Scale_Factor
    ML_Model.fit(Total_Train_2D[:,:-1],Total_Train_2D[:,-1])
    Center_Concate_2D[:,-3:]=Center_Concate_2D[:,-3:]/Scale_Factor
    L_Center_Pure_PreRes=ML_Model.predict(Center_Concate_2D[:,:-1])

    'step 12'
    '''
    L_Center_QA=Patch_L_Flag_Pad_Target_1D
    L_Center_QA[np.isnan(Patch_L_NDVI_Mixed)]=2
    '''
    #L_Center_QA=Patch_L_Flag_Pad_Target_1D
    L_Center_Pure_PreRes[L_Center_Pure_PreRes<Valid_Min]=Valid_Min
    L_Center_Pure_PreRes[L_Center_Pure_PreRes>Valid_Max]=Valid_Max
    Patch_L_NDVI_Mixed[np.isnan(Patch_L_NDVI_Mixed)]=L_Center_Pure_PreRes[np.isnan(Patch_L_NDVI_Mixed)]
    
    Patch_L_NDVI_Smooth=WWHD_Smooth((Patch_L_NDVI_Mixed,L_Center_QA))

    if Patch_ID>=9999 and (Patch_ID+1)%10000==0:
        Batch_Num=int((Patch_ID+1)/10000)
        Batch_Time=datetime.datetime.now()
        print("the finished time of the {batch_num} batch(10000) prediction is {batch_time}".format(batch_num=Batch_Num,batch_time=Batch_Time))    
    
    #return Patch_L_NDVI_Mixed,L_Center_Pure_PreRes,Patch_L_NDVI_Smooth
    return Patch_L_NDVI_Smooth


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    Fusion_Pools=MPIPool()
    if not Fusion_Pools.is_master():
        Fusion_Pools.wait()
        sys.exit(0)
    
    "step 01: File Path"
    Sample_L578_NDVI_File_Path=sys.argv[5]
    Sample_MODIS_NDVI_File_Path=sys.argv[6]
    Sample_ERA5_Meteo_File_Path=sys.argv[7]
    Sample_NDVI_Fusion_File_Path=sys.argv[8]

    "step 02: Load and Preprocess Data"
    # 2.1 landsat and modis data
    Sample_L578_NDVI_3D=gdal.Open(glob.glob(Sample_L578_NDVI_File_Path+Sample_Name+'*.tif')[0]).ReadAsArray()
    Sample_MODIS_NDVI_3D=gdal.Open(glob.glob(Sample_MODIS_NDVI_File_Path+Sample_Name+'*.tif')[0]).ReadAsArray()
    Sample_L578_NDVI_3D[Sample_L578_NDVI_3D>1000]=Cloud_ID
    Sample_L578_NDVI_3D[Sample_L578_NDVI_3D<=0]=Cloud_ID
    Sample_L578_NDVI_3D=Landsat_Imgs_PreProcess(Sample_L578_NDVI_3D,Chunk_Size,Cloud_ID)

    # 2.2 meteo data
    Samples_Meteo_Dirs=os.listdir(Sample_ERA5_Meteo_File_Path)
    Sample_Meteo_Dir=[meteo_dir for meteo_dir in Samples_Meteo_Dirs if Sample_Name in meteo_dir][0]
    Sample_ERA5_TA_8Day_tifs=natsort.natsorted(glob.glob(Sample_ERA5_Meteo_File_Path+Sample_Meteo_Dir+'/*Temperature*.tif'))
    Sample_ERA5_SR_8Day_tifs=natsort.natsorted(glob.glob(Sample_ERA5_Meteo_File_Path+Sample_Meteo_Dir+'/*Solar*.tif'))
    Sample_ERA5_VPD_8Day_tifs=natsort.natsorted(glob.glob(Sample_ERA5_Meteo_File_Path+Sample_Meteo_Dir+'/*VPD*.tif'))

    Sample_ERA5_TA_8Day_3D=np.vstack([gdal.Open(ta_tif).ReadAsArray() for ta_tif in Sample_ERA5_TA_8Day_tifs])
    Sample_ERA5_SR_8Day_3D=np.vstack([gdal.Open(sr_tif).ReadAsArray() for sr_tif in Sample_ERA5_SR_8Day_tifs])
    Sample_ERA5_VPD_8Day_3D=np.vstack([gdal.Open(vpd_tif).ReadAsArray() for vpd_tif in Sample_ERA5_VPD_8Day_tifs])

    Start_ID_Valid=(Select_Start_Year-Start_Year)*Batch_Slices
    End_ID_Valid=(Select_End_Year-Start_Year+1)*Batch_Slices

    Sample_Landsat_NDVI_Select_3D=Sample_L578_NDVI_3D[Start_ID_Valid:End_ID_Valid]
    Sample_MODIS_NDVI_Select_3D=Sample_MODIS_NDVI_3D[Start_ID_Valid:End_ID_Valid]

    Sample_ERA5_TA_8Day_Select_3D=Sample_ERA5_TA_8Day_3D[Start_ID_Valid:End_ID_Valid]
    Sample_ERA5_SR_8Day_Select_3D=Sample_ERA5_SR_8Day_3D[Start_ID_Valid:End_ID_Valid]
    Sample_ERA5_VPD_8Day_Select_3D=Sample_ERA5_VPD_8Day_3D[Start_ID_Valid:End_ID_Valid]

    "step 03: Set Parameters"
    Time_Len,Sample_Rows,Sample_Cols=Sample_Landsat_NDVI_Select_3D.shape
    Patches_Extent01,Patches_Extent02=PatchesExtents(Sample_Rows,Sample_Cols,Patch_Size,Win_R)
    L578_NDVI_PatchesPad_List=PatchesPad(Patches_Extent02,Sample_Landsat_NDVI_Select_3D[None,],Patch_Size,Win_R)
    MODIS_NDVI_PatchesPad_List=PatchesPad(Patches_Extent02,Sample_MODIS_NDVI_Select_3D[None,],Patch_Size,Win_R)

    ERA5_TA_PatchesPad_List=PatchesPad(Patches_Extent02,Sample_ERA5_TA_8Day_Select_3D[None,],Patch_Size,Win_R)
    ERA5_SR_PatchesPad_List=PatchesPad(Patches_Extent02,Sample_ERA5_SR_8Day_Select_3D[None,],Patch_Size,Win_R)
    ERA5_VPD_PatchesPad_List=PatchesPad(Patches_Extent02,Sample_ERA5_VPD_8Day_Select_3D[None,],Patch_Size,Win_R)

    "step 04: Summary Data" 
    Patches_ID_List=list(np.arange(0,len(L578_NDVI_PatchesPad_List)))
    PatchesPad_ID_Meteo_MODIS_Landsat_List=list(zip(Patches_ID_List,ERA5_TA_PatchesPad_List,ERA5_SR_PatchesPad_List,
                                                    ERA5_VPD_PatchesPad_List,MODIS_NDVI_PatchesPad_List,
                                                    L578_NDVI_PatchesPad_List))
    del ERA5_TA_PatchesPad_List,ERA5_SR_PatchesPad_List,ERA5_VPD_PatchesPad_List,\
        MODIS_NDVI_PatchesPad_List,L578_NDVI_PatchesPad_List,Patches_ID_List
    gc.collect()

    "step 05: Parallal Fusion"
    Para_Time_0=time.time()
    print("the start time of parallel execute:{t0}".format(t0=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #InENVI_Fusion_List=Fusion_Pools.map(InENVI,PatchesPad_ID_Meteo_MODIS_Landsat_List)
    InENVI_Fusion_List=Fusion_Pools.map(InENVI_Normal_V2,PatchesPad_ID_Meteo_MODIS_Landsat_List)
    
    Para_Time_1=time.time()
    print('the end time of parallel execute:{t1},the total running time:{T:.2f}h'.format(t1=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),T=(Para_Time_1-Para_Time_0)/3600))
    del PatchesPad_ID_Meteo_MODIS_Landsat_List
    gc.collect()

    "step 06: Post Process"
    if Win_R<10:
        Win_R_Str='WinR0'+str(Win_R)
    elif Win_R<100:
        Win_R_Str='WinR'+str(Win_R)
    else:
        pass
    
    InENVI_Fusion_2D=np.array(InENVI_Fusion_List)
    InENVI_Fusion_3D=InENVI_Fusion_2D.T.reshape(Time_Len,-1,Sample_Cols)
    InENVI_Fusion_3D_Scale=InENVI_Fusion_3D*Scale_Factor
    InENVI_Fusion_3D_Scale_Int16=InENVI_Fusion_3D_Scale.astype(np.int16)

    np.save(Sample_NDVI_Fusion_File_Path+Sample_Name+'_NDVI_InENVIV2_Fusion_8Day_Normal_Simu_WGS30m_Y_'+str(Select_Start_Year)+'_'+str(Select_End_Year)+'_'+Win_R_Str+'_Int16.npy',InENVI_Fusion_3D_Scale_Int16)













