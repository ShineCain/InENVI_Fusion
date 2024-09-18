
import numpy as np
from Model import ModelSelect
from sklearn.model_selection import KFold
from WWHD import WWHD_Smooth

def Cross_Valid_Modify(Center_2D,Similar_3D,Folds_N,Cloud_ID,Scale_Factor,Selected_Model='LGB'):
    'step 01'
    K_Folds=KFold(n_splits=Folds_N)
    Center_Test_PredRes_1D=np.array([])
    
    'step 02'
    for train_cv_id,test_cv_id in K_Folds.split(Center_2D):
        #Center_L_True_TS=np.copy(Center_2D[:,-1]/Scale_Factor)

        'step 2.1'
        Similar_3D_Copy=np.copy(Similar_3D)
        Center_Train_2D=np.copy(Center_2D[train_cv_id])
        Center_Test_2D=np.copy(Center_2D[test_cv_id])
        Similar_List=[]

        'step 2.2'
        for simi_2d in Similar_3D_Copy:
            simi_2d[test_cv_id,-1]=Cloud_ID
            Similar_List.append(simi_2d)

        Similar_3D_Copy=np.array(Similar_List)
        Similar_2D=np.vstack([similar_data for similar_data in Similar_3D_Copy])
        Similar_Train_Valid_2D=Similar_2D[Similar_2D[:,-1]!=Cloud_ID]

        'step 2.3'
        Center_Train_Valid_2D=Center_Train_2D[Center_Train_2D[:,-1]!=Cloud_ID]
        Total_Train_Valid_2D=np.concatenate((Similar_Train_Valid_2D,Center_Train_Valid_2D),axis=0)

        Total_Train_Valid_2D[:,-3:]=Total_Train_Valid_2D[:,-3:]/Scale_Factor
        Center_Test_2D[:,-3:]=Center_Test_2D[:,-3:]/Scale_Factor

        'step 2.4'
        ML_Model=ModelSelect(Selected_Model)
        ML_Model.fit(Total_Train_Valid_2D[:,:-1],Total_Train_Valid_2D[:,-1])
        Center_Test_PredRes_Tmp=ML_Model.predict(Center_Test_2D[:,:-1])
        Center_Test_PredRes_1D=np.append(Center_Test_PredRes_1D,Center_Test_PredRes_Tmp)
    
    #Center_Test_PredRes_Sig=Sigmoid(Center_Test_PredRes_1D)
    '''
    X=sm.add_constant(Center_Test_PredRes_1D[Center_2D[:,-1]!=Cloud_ID])
    LM=sm.OLS(Center_2D[:,-1][Center_2D[:,-1]!=Cloud_ID]/Scale_Factor,X).fit()
    KB=LM.params
    Center_Test_PredRes_Adj=KB[1]*Center_Test_PredRes_1D+KB[0]
    #Center_Test_PredRes_1D=np.array(Center_Test_PredRes_List).ravel()
    print(KB)
    '''
    return Center_Test_PredRes_1D

def CV_Spec_Rth0(Center_2D,Annual_Curve_Use_Flag,Similar_3D,Center_QA,Folds_N,Spec_Year,Cloud_ID,Scale_Factor,Select_Start_Year,Batch_Slice=46,Selected_Model='LGB'):
    '''
    Spec Year: for example 2017
    Cross Valid:Radius>0
    '''
    'step 01'
    K_Folds=KFold(n_splits=Folds_N)
    #Center_Spec_Test_PredRes_1D=np.array([])
    
    'step 02'
    Spec_ID0,Spec_ID1=(Spec_Year-Select_Start_Year)*Batch_Slice,(Spec_Year-Select_Start_Year+1)*Batch_Slice
    
    #Records_ID=np.arange(0,len(Center_2D_Copy))
    #Ref_Rel_ID0,Ref_Rel_ID1=(Spec_Year-Select_Start_Year)*Batch_Slice,(Spec_Year-Select_Start_Year+1)*Batch_Slice
    #Center_Other_2D=Center_2D_Copy[~np.in1d(Records_ID,np.arange(Spec_ID0,Spec_ID1))] # Other years records
    #Center_Other_Test=Center_Other_2D[Center_Other_2D[:,-1]==Cloud_ID]
    #Center_Other_Test[:,-3:]=Center_Other_Test[:,-3:]/Scale_Factor

    'step 03'
    Center_Spec_2D=np.copy(Center_2D[Spec_ID0:Spec_ID1])
    Patch_L_NDVI_Pad_Target_1D=np.copy(Center_2D[:,-1]).astype(float)
    Patch_L_NDVI_Pad_Target_1D[Patch_L_NDVI_Pad_Target_1D==Cloud_ID]=np.nan
    Valid_Min,Valid_Max=np.nanmin(Patch_L_NDVI_Pad_Target_1D)/Scale_Factor,np.nanmax(Patch_L_NDVI_Pad_Target_1D)/Scale_Factor
    
    L_Center_Smooth_CV_1D=np.array([])
    for train_cv_id,test_cv_id in K_Folds.split(Center_Spec_2D):
        'step 3.1'
        #Center_L_True_TS=np.copy(Center_2D[:,-1]/Scale_Factor)
        #Center_Spec_QA[test_cv_id]
        Center_2D_Copy,Similar_3D_Copy=np.copy(Center_2D),np.copy(Similar_3D)
        Center_QA_Copy=np.copy(Center_QA)
        
        #Center_Spec_QA=Center_QA_Copy[Spec_ID0:Spec_ID1]
        Similar_Spec_3D=np.copy(Similar_3D[:,Spec_ID0:Spec_ID1])
        L_Smooth_PredRes_Tmp=np.copy(Center_2D[:,-1])/Scale_Factor

        'step 3.2'
        #Similar_Spec_3D_Copy=np.copy(Similar_Spec_3D)
        Center_Spec_2D_Copy=np.copy(Center_Spec_2D)
        Center_Spec_2D_Copy[test_cv_id,-1]=Cloud_ID
        Center_2D_Copy[Spec_ID0:Spec_ID1]=Center_Spec_2D_Copy
        
        #Center_Spec_Train_2D=np.copy(Center_Spec_2D[train_cv_id])
        #Center_Spec_Test_2D=np.copy(Center_Spec_2D[test_cv_id])

        'step 3.3'
        Similar_Spec_List=[]
        for simi_spec_2d in Similar_Spec_3D:
            simi_spec_2d[test_cv_id,-1]=Cloud_ID
            Similar_Spec_List.append(simi_spec_2d)

        Similar_Spec_3D_Copy=np.array(Similar_Spec_List)
        Similar_3D_Copy[:,Spec_ID0:Spec_ID1]=Similar_Spec_3D_Copy

        Similar_2D=np.vstack([similar_data for similar_data in Similar_3D_Copy])
        Similar_Train_Valid_2D=Similar_2D[Similar_2D[:,-1]!=Cloud_ID]

        'step 3.4'
        Center_QA_Copy[Center_2D_Copy[:,-1]==Cloud_ID]=2
        Center_Train_Valid_2D=Center_2D_Copy[Center_2D_Copy[:,-1]!=Cloud_ID]
        Center_Train_InValid_2D=Center_2D_Copy[Center_2D_Copy[:,-1]==Cloud_ID] # include the test id
        
        Total_Train_Valid_2D=np.concatenate((Similar_Train_Valid_2D,Center_Train_Valid_2D),axis=0)
        
        if Annual_Curve_Use_Flag==1:
            Total_Train_Valid_2D[:,-3:]=Total_Train_Valid_2D[:,-3:]/Scale_Factor
            #Center_Spec_Test_2D[:,-3:]=Center_Spec_Test_2D[:,-3:]/Scale_Factor
            Center_Train_InValid_2D[:,-3:]=Center_Train_InValid_2D[:,-3:]/Scale_Factor
        else:
            Total_Train_Valid_2D[:,-2:]=Total_Train_Valid_2D[:,-2:]/Scale_Factor
            #Center_Spec_Test_2D[:,-3:]=Center_Spec_Test_2D[:,-3:]/Scale_Factor
            Center_Train_InValid_2D[:,-2:]=Center_Train_InValid_2D[:,-2:]/Scale_Factor

        'step 3.5'
        ML_Model=ModelSelect(Selected_Model)
        ML_Model.fit(Total_Train_Valid_2D[:,:-1],Total_Train_Valid_2D[:,-1])
        L_Smooth_PredRes_Tmp_QA2=ML_Model.predict(Center_Train_InValid_2D[:,:-1]) # include the test id
        L_Smooth_PredRes_Tmp_QA2[L_Smooth_PredRes_Tmp_QA2<Valid_Min]=Valid_Min
        L_Smooth_PredRes_Tmp_QA2[L_Smooth_PredRes_Tmp_QA2>Valid_Max]=Valid_Max
        L_Smooth_PredRes_Tmp[Center_QA_Copy==2]=L_Smooth_PredRes_Tmp_QA2

        L_Smooth_PredRes_Tmp=WWHD_Smooth((L_Smooth_PredRes_Tmp,Center_QA_Copy))
        L_Center_Smooth_CV_1D=np.append(L_Center_Smooth_CV_1D,L_Smooth_PredRes_Tmp[Spec_ID0:Spec_ID1][test_cv_id])

        #L_Center_Other_Test_PredRes_Tmp=ML_Model.predict(Center_Other_Test[:,:-1])
        
        #Center_Spec_Test_PredRes_Tmp=ML_Model.predict(Center_Spec_Test_2D[:,:-1])
        #Center_Spec_Test_PredRes_1D=np.append(Center_Spec_Test_PredRes_1D,Center_Spec_Test_PredRes_Tmp)

    
    #Center_Test_PredRes_Sig=Sigmoid(Center_Test_PredRes_1D)
    '''
    X=sm.add_constant(Center_Test_PredRes_1D[Center_2D[:,-1]!=Cloud_ID])
    LM=sm.OLS(Center_2D[:,-1][Center_2D[:,-1]!=Cloud_ID]/Scale_Factor,X).fit()
    KB=LM.params
    Center_Test_PredRes_Adj=KB[1]*Center_Test_PredRes_1D+KB[0]
    #Center_Test_PredRes_1D=np.array(Center_Test_PredRes_List).ravel()
    print(KB)
    '''
    #return Center_Test_PredRes_1D

    return L_Center_Smooth_CV_1D

def CV_Spec_Req0(Center_2D,Annual_Curve_Use_Flag,Center_QA,Folds_N,Spec_Year,Cloud_ID,Scale_Factor,Select_Start_Year,Batch_Slice=46,Selected_Model='LGB'):
    '''
    Spec Year:for example 2017
    Cross Valid:Radius equals 0
    '''
    'step 01'
    K_Folds=KFold(n_splits=Folds_N)
    
    'step 02'
    Spec_ID0,Spec_ID1=(Spec_Year-Select_Start_Year)*Batch_Slice,(Spec_Year-Select_Start_Year+1)*Batch_Slice

    'step 03'
    Center_Spec_2D=np.copy(Center_2D[Spec_ID0:Spec_ID1])
    Patch_L_NDVI_Pad_Target_1D=np.copy(Center_2D[:,-1].astype(float))
    Patch_L_NDVI_Pad_Target_1D[Patch_L_NDVI_Pad_Target_1D==Cloud_ID]=np.nan
    Valid_Min,Valid_Max=np.nanmin(Patch_L_NDVI_Pad_Target_1D)/Scale_Factor,np.nanmax(Patch_L_NDVI_Pad_Target_1D)/Scale_Factor
    
    L_Center_Smooth_CV_1D=np.array([])

    for train_cv_id,test_cv_id in K_Folds.split(Center_Spec_2D):
        #Center_L_True_TS=np.copy(Center_2D[:,-1]/Scale_Factor)

        'step 3.1'
        Center_2D_Copy,Center_QA_Copy=np.copy(Center_2D),np.copy(Center_QA)
        L_Smooth_PredRes_Tmp=np.copy(Center_2D[:,-1])/Scale_Factor

        'step 3.2'
        #Similar_Spec_3D_Copy=np.copy(Similar_Spec_3D)
        Center_Spec_2D_Copy=np.copy(Center_Spec_2D)
        Center_Spec_2D_Copy[test_cv_id,-1]=Cloud_ID
        Center_2D_Copy[Spec_ID0:Spec_ID1]=Center_Spec_2D_Copy

        'step 3.3'
        Center_QA_Copy[Center_2D_Copy[:,-1]==Cloud_ID]=2
        Center_Train_Valid_2D=Center_2D_Copy[Center_2D_Copy[:,-1]!=Cloud_ID]
        Center_Train_InValid_2D=Center_2D_Copy[Center_2D_Copy[:,-1]==Cloud_ID] # include the test id

        if Annual_Curve_Use_Flag==1:
            Center_Train_Valid_2D[:,-3:]=Center_Train_Valid_2D[:,-3:]/Scale_Factor
            Center_Train_InValid_2D[:,-3:]=Center_Train_InValid_2D[:,-3:]/Scale_Factor
        else:
            Center_Train_Valid_2D[:,-2:]=Center_Train_Valid_2D[:,-2:]/Scale_Factor
            Center_Train_InValid_2D[:,-2:]=Center_Train_InValid_2D[:,-2:]/Scale_Factor

        'step 3.4'
        ML_Model=ModelSelect(Selected_Model)
        ML_Model.fit(Center_Train_Valid_2D[:,:-1],Center_Train_Valid_2D[:,-1])
        L_Center_Smooth_PredRes_Tmp_QA2=ML_Model.predict(Center_Train_InValid_2D[:,:-1]) # include the test id
        L_Center_Smooth_PredRes_Tmp_QA2[L_Center_Smooth_PredRes_Tmp_QA2<Valid_Min]=Valid_Min
        L_Center_Smooth_PredRes_Tmp_QA2[L_Center_Smooth_PredRes_Tmp_QA2>Valid_Max]=Valid_Max

        'step 3.5'
        L_Smooth_PredRes_Tmp[Center_QA_Copy==2]=L_Center_Smooth_PredRes_Tmp_QA2
        L_Smooth_PredRes_Tmp=WWHD_Smooth((L_Smooth_PredRes_Tmp,Center_QA_Copy))
        L_Center_Smooth_CV_1D=np.append(L_Center_Smooth_CV_1D,L_Smooth_PredRes_Tmp[Spec_ID0:Spec_ID1][test_cv_id])

    return L_Center_Smooth_CV_1D


       



