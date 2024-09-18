
import numpy as np
from sklearn.metrics import mean_squared_error

def Simu_Accu_Cal(Simu_3D,True_3D,Mask_3D):
    
    R2_1D,RMSE_1D=np.array([]),np.array([])
    Time_Len=len(Simu_3D)
    
    for i in range(Time_Len):
        Simu_Tmp,True_Tmp,Mask_Tmp=Simu_3D[i],True_3D[i],Mask_3D[i]
        R2_Tmp=np.corrcoef(Simu_Tmp[Mask_Tmp==1].ravel(),True_Tmp[Mask_Tmp==1].ravel())[0,1]**2
        
        Valid_Num=np.sum(Mask_Tmp)
        if Valid_Num>2:
            RMSE_Tmp=np.sqrt(mean_squared_error(Simu_Tmp[Mask_Tmp==1].ravel(),True_Tmp[Mask_Tmp==1].ravel())) # simu flag:1
        else:
            RMSE_Tmp=np.nan
        #R2_List.append(R2_Tmp)
        #RMSE_List.append()
        R2_1D=np.append(R2_1D,R2_Tmp)
        RMSE_1D=np.append(RMSE_1D,RMSE_Tmp)
    
    #R2_1D=np.array(R2_List)
    return R2_1D,RMSE_1D
