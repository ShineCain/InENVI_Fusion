import numpy as np
import scipy.interpolate as spi
from scipy.signal import savgol_filter

'''
LACF: Landsat Annual Curve Fill
'''

def LACF(L_Pixel_Annual_Ave_DOY,Batch_Slices=46,Cloud_ID=-9):
    '''
    Pixel_L_Annual_DOY_TS:have been scaled(必须是介于0~1的float64位数据)
    '''
    Pixel_L_Annual_DOY_TS_Copy=L_Pixel_Annual_Ave_DOY.copy().astype(float)
    Pixel_Annual_TS_Nan_Loc=np.argwhere(np.isnan(L_Pixel_Annual_Ave_DOY))[:,0]
    #Pixel_Annual_QA=np.zeros(len(Pixel_L_Annual_DOY_TS))

    Time_ID_Set=set(list(range(len(L_Pixel_Annual_Ave_DOY))))
    Time_Nan_Loc_Set=set(list(Pixel_Annual_TS_Nan_Loc))
    Time_ID_X=np.sort(np.array(list(Time_ID_Set-Time_Nan_Loc_Set)))
    
    if len(Time_ID_X)>=int(Batch_Slices/2):
        Interplo_fit=spi.CubicSpline(Time_ID_X,L_Pixel_Annual_Ave_DOY[Time_ID_X])
        Pixel_L_Annual_DOY_TS_Copy[list(Time_Nan_Loc_Set)]=Interplo_fit(list(Time_Nan_Loc_Set))
        #Pixel_Annual_QA[Pixel_Annual_TS_Nan_Loc]=2
    else:
        Filled_Series=np.random.randint(10,size=len(Pixel_L_Annual_DOY_TS_Copy))/10
        Pixel_L_Annual_DOY_TS_Copy[:]=Cloud_ID+Filled_Series
        
    return Pixel_L_Annual_DOY_TS_Copy

def LACF_Smooth(L_Pixel_Annual_Ave_DOY,Batch_Slices=46,Cloud_ID=-9):
    '''
    Pixel_L_Annual_DOY_TS:have been scaled(必须是介于0~1的float64位数据)
    '''
    Pixel_L_Annual_DOY_TS_Copy=L_Pixel_Annual_Ave_DOY.copy().astype(float)
    Pixel_Annual_TS_Nan_Loc=np.argwhere(np.isnan(L_Pixel_Annual_Ave_DOY))[:,0]
    #Pixel_Annual_QA=np.zeros(len(Pixel_L_Annual_DOY_TS))

    Time_ID_Set=set(list(range(len(L_Pixel_Annual_Ave_DOY))))
    Time_Nan_Loc_Set=set(list(Pixel_Annual_TS_Nan_Loc))
    Time_ID_X=np.sort(np.array(list(Time_ID_Set-Time_Nan_Loc_Set)))
    
    if len(Time_ID_X)>=int(Batch_Slices/2):
        Interplo_fit=spi.CubicSpline(Time_ID_X,L_Pixel_Annual_Ave_DOY[Time_ID_X])
        Pixel_L_Annual_DOY_TS_Copy[list(Time_Nan_Loc_Set)]=Interplo_fit(list(Time_Nan_Loc_Set))
        #Pixel_Annual_QA[Pixel_Annual_TS_Nan_Loc]=2
    else:
        Filled_Series=np.random.randint(10,size=len(Pixel_L_Annual_DOY_TS_Copy))/10
        Pixel_L_Annual_DOY_TS_Copy[:]=Cloud_ID+Filled_Series #有些问题
    return Pixel_L_Annual_DOY_TS_Copy
     

