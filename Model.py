import sys
import xgboost as xgb
import lightgbm as lgb

from sklearn.svm import SVR
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor,GradientBoostingRegressor,RandomForestRegressor

def ModelSelect(ML_Model_Name):
    if ML_Model_Name=="XGB":
        xgb_model_params={'objective':'reg:logistic','eval_metric':'rmse','booster':'gbtree','random_state':36,
                          'subsample':0.8,'reg_lambda':1.1,'seed':0,'n_estimators':400,'max_depth':6,'max_delta_step':6,
                          'min_child_weight':0.3,'eta':0.1,'n_jobs':1}
        selected_model=xgb.XGBRegressor(**xgb_model_params)
    
    elif ML_Model_Name=="LGB":
        lgb_model_params={'objective':'regression','metric':'rmse','learning_rate':0.05,'random_state':10,
                          'subsample':0.8,'reg_lambda':1.1,'n_estimators':300,'max_depth':5,'max_delta_step':6,
                          'max_depth':5,'num_leaves':30,'verbose':-1,'n_jobs':1}
        selected_model=lgb.LGBMRegressor(**lgb_model_params)
        
    elif ML_Model_Name=="RFR":
        rfr_model_params={'n_estimators':400,'max_depth':6,'verbose':0,'criterion':'mse','random_state':35,'n_jobs':1}
        RFR_Model=RandomForestRegressor(**rfr_model_params)
        selected_model=RFR_Model
        
    elif ML_Model_Name=="MLP":
        MLP_Reg_Model=MLPRegressor(hidden_layer_sizes=(30,15),activation='relu',solver='adam',
                                   alpha=0.05,learning_rate_init=0.05,learning_rate='adaptive',
                                   max_iter=3600,random_state=1)
        selected_model=MLP_Reg_Model
        
    elif ML_Model_Name=="SVM":
        SVM_Reg_Model=SVR(kernel='rbf',C=1)
        selected_model=SVM_Reg_Model
        
    elif ML_Model_Name=="MLR":
        MLR_Model=LinearRegression(n_jobs=1) #多元线性回归
        selected_model=MLR_Model
    
    elif ML_Model_Name=="CAT":
        Cat_Model=CatBoostRegressor(n_estimators=400,depth=6,learning_rate=0.05,
                                    loss_function='RMSE',l2_leaf_reg=3,verbose=False,thread_count=1,
                                    random_state=6)
        selected_model=Cat_Model
    
    elif ML_Model_Name=="ADA":
        Ada_Model=AdaBoostRegressor(n_estimators=400,learning_rate=0.05,random_state=18)
        selected_model=Ada_Model
    
    elif ML_Model_Name=="GBR":
        GBR_Model=GradientBoostingRegressor(n_estimators=400,max_depth=6,learning_rate=0.05,random_state=16)
        selected_model=GBR_Model
    
    elif ML_Model_Name=="BR":
        BR_Model=BaggingRegressor(n_estimators=400,n_jobs=1,random_state=6)
        selected_model=BR_Model
    
    elif ML_Model_Name=="DTR":
        DTR_Model=DecisionTreeRegressor(max_depth=6,random_state=8) # 表现差不予考虑
        selected_model=DTR_Model
    
    elif ML_Model_Name=="KNN":
        KNN_Model=KNeighborsRegressor(n_neighbors=6,n_jobs=1)
        selected_model=KNN_Model
        
    else:
        print('the machine learning model name is invalid,please input one of {"XGB","RFR","MLP","SVM","MLR","ADA","CAT","GBR","BR","DTR","KNN"}')
        sys.exit()
    
    return selected_model

