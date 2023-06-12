import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import cx_Oracle
import pickle
#import db_config
import datetime
import itertools
import warnings
from django.db import connection 


def displayText(val1,val2):
    res = val1+val2 
    

    timestart = datetime.datetime.now()
    # con = cx_Oracle.connect(db_config.username, db_config.password, db_config.dsn)
    # print("Database version:", con.version)
    dfs_query='select * from (select * from (select py_customer_id, customer_id, customer_bank_name, customer_bank_short_name, customer_type, customer_title, customer_name, customer_creation_date, status_code as status_code_mst , customer_status_date, cust_create_from, date_of_birth, customer_age, customer_gender, cust_birth_place, martial_status, education_id, education_desc, const_code, const_desc, occupation_code, occupation_desc, community_code, community_desc, caste, caste_code, nationality_code, nationality_desc, domicile_code, domicile_desc, mobile_no, email_id, aadhaar_card, blood_group, risk_type_id, risk_type_desc, member_flag, pan, tan, form_60_flag, credit_card_flag, debit_card_flag, sms_banking_flag, mobile_banking_flag, net_banking_flag, occupation_details, employed_with, annual_income, guarantee_count, sureity_count, sureity_amount, dmat_accno, nre_nro_nri_flag, permanent_return_date, identification_id, identification_details, issue_date, issued_by, address_proof_id, address_proof_details, documents_given, kyc_complete_flag, kyc_complete_date, kyc_number, establish_licence_flag, establish_licence_no, establish_licence_date, establish_licence_exp, gst_no, gst_reg_date, gst_verify_flag, gst_verify_date, office_mst_id, old_primary_key, permanent_add, official_add, resisdential_add from py_customer_mast) m inner join py_cust_general g on m.py_customer_id=g.py_customer_id inner join py_customer_all_loans l on m.py_customer_id=l.py_customer_id)'#' where PY_CUSTOMER_ID = {}'.format(py_customer_id)
    df=pd.read_sql_query(dfs_query,connection)
    df = df.loc[:,~df.columns.duplicated()]
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)

    from db_function import Report_Data
    df2 = Report_Data(df,connection)
    #con.close()

    df= df2.drop(['CUSTOMER_ID','CUSTOMER_BANK_NAME','CUSTOMER_CREATION_DATE','CUSTOMER_TITLE','CUSTOMER_NAME','STATUS_CODE_MST',
                'CUSTOMER_STATUS_DATE','DATE_OF_BIRTH','CUST_BIRTH_PLACE','EDUCATION_DESC','CONST_DESC',
            'OCCUPATION_DESC','COMMUNITY_DESC','CASTE','NATIONALITY_DESC','DOMICILE_DESC','EMAIL_ID','RISK_TYPE_DESC','TAN',
                'OCCUPATION_DETAILS','EMPLOYED_WITH','DMAT_ACCNO','PERMANENT_RETURN_DATE','IDENTIFICATION_DETAILS','ISSUE_DATE',
            'ISSUED_BY','ADDRESS_PROOF_DETAILS','DOCUMENTS_GIVEN','KYC_COMPLETE_DATE','KYC_NUMBER','ESTABLISH_LICENCE_NO',
                'ESTABLISH_LICENCE_DATE','ESTABLISH_LICENCE_EXP','GST_NO','GST_REG_DATE','GST_VERIFY_DATE','OFFICE_MST_ID',
                'OLD_PRIMARY_KEY','OFFICIAL_ADD','RESISDENTIAL_ADD','PERMANENT_ADD','PY_CUST_GEN_MST_ID',
            'FOREGIN_TOUR_YN','CUST_FRDT_IN_MTHS','SUREITY_TO_NOOFAC', 'PY_CUST_MST_ID_ALL_LOANS','PY_CUST_MST_ID_ALL',
            'PY_CUSTCCOD_MST_ID','PY_CUSTLN_MST_ID','GLTITLE','NEW_AC_NUMBER','ACCOUNT_NAME','ACCOUNT_OPEN_DATE',
            'ACCOUNT_STATUS_DATE','DP_AMOUNT','EXPIRY_DATE','LAST_INTT_RECEIVED_DT','LIEN_DATE','NPA_DATE',
                'SANCTION_DATE','SECURITY_CODE','VALUE_DESCRIPTION','LOAN_TYPE','BLOOD_GROUP','GLCODE'],axis = 1)

    from db_function import Mobile_adhar_pan_flag
    df_cat = df.select_dtypes(include = 'object').copy()
    df_num = df.select_dtypes(include = 'number').copy()
    df_cat = Mobile_adhar_pan_flag(df_cat)
    from db_function import Missing_Treatment
    df = Missing_Treatment(df_num,df_cat)

    df['good_bad'] = np.where(df.loc[:, 'NPA_CLASS_ID'].isin(['ST','SS']), 1, 0)
    # Drop the original column
    df = df.drop(columns = ['NPA_CLASS_ID'])
    X = df.drop('good_bad', axis = 1)
    y = df['good_bad']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100, stratify = y)
    train= df.loc[df['STATUS_CODE'] == 'Closed']
    X_train = train.drop('good_bad', axis = 1)
    y_train = train['good_bad']
    test= df.loc[df['STATUS_CODE'] != 'Closed']
    X_test = test.drop('good_bad', axis = 1)
    y_test = test['good_bad']

    from db_function import MultiColumnLabelEncoder
    df_cat = df.select_dtypes(include = 'object').copy()    
    cat_cols = list(df_cat.columns)
    X1=MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X)
    X_train1=MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X_train)
    X_test1=MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X_test)
    X_train = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X_train)
    X_test = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X_test)
    # reindex the dummied test set variables to make sure all the feature columns in the train set are also available in the test set
    X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)
    X4 = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X)
    X3 = X4.reindex(labels=X4.columns, axis=1, fill_value=0)
    df_num = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X)

    from db_function import Anova
    anova = Anova(df_num,y)
    X_train2 = X_train[anova]
    X_test2 = X_test[anova]

    from db_function import woe_ordered_continuous
    from db_function import Numerical_Features_Range
    X_train_prepr = X_train2.copy()
    y_train_prepr = y_train.copy()
    df_temp3 = Numerical_Features_Range(X_train_prepr,y_train_prepr,anova)

    cols1 = pd.DataFrame(list(itertools.chain(*df_temp3)))
    cols1.to_csv(r'C:\Users\vishal.lote\Documents\My Received Files\PY-SAURABH DUBEY (LAPTOP)\credit_score\TDataset\file.csv')

    cols = list(itertools.chain(*df_temp3))
    cols2 = pd.DataFrame(cols)
    cols1 = cols2[0].str.split(':').str[0]        
    mylist = cols2[0].str.split(':').str[1]
    cols_list = list(X_train.columns)
    class WoE_Binning(BaseEstimator, TransformerMixin):
        def __init__(self, X): # no *args or *kargs
            self.X = X
        def fit(self, X, y = None):
            return self #nothing else to do
        def transform(self, X):
            X_new = X.loc[:,X.columns!=cols_list]
            for i in range (len(mylist)):
                X_new[cols[i]] = np.where((X[cols1[i]] > float(mylist[i].split(',')[0])) & (X[cols1[i]] <= float(mylist[i].split(',')[1])), 1, 0)
            return X_new

    reg = LogisticRegression(max_iter=500, class_weight = 'balanced',verbose=0)
    woe_transform = WoE_Binning(X)
    pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])
    pipeline = pipeline.fit(X_train, y_train)

    pd.options.mode.chained_assignment = None  # default='warn'
    accr=[]
    y_pred_log = pipeline.predict(X_test)
    accuracy_score(y_test,y_pred_log)
    accr.append(round(accuracy_score(y_test,y_pred_log),4)*100)

    from db_function import Training_woe_transform
    summary_table = Training_woe_transform(X_train,woe_transform,pipeline)

    from db_function import prediction_testing_set
    y_test_proba = prediction_testing_set(X_test,y_test,pipeline)

    # # Enter input Py_customer_id for check the output on individual ID
    # py_customer_id=int(input('Enter the customer id: '))
    # df2 = df2.loc[df2['PY_CUSTOMER_ID'] == py_customer_id]
    # df2 = df2.reset_index(drop=True)
    # X1 = X1.loc[X1['PY_CUSTOMER_ID'] == py_customer_id]
    # X12 = X3.copy()
    # X12 = X12.loc[X12['PY_CUSTOMER_ID'] == py_customer_id]

    tr = 0.5
    y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr, 1, 0)
    confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'])
    fpr, tpr, thresholds = roc_curve(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])

    df_scorecard = summary_table
    df_scorecard.reset_index(inplace = True)
    df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
        
    min_score = 300
    max_score = 850
    min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
    max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
    df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
    df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0,'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
    df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
    print(df_scorecard['Score - Preliminary'])
    df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
    df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
    print(df_scorecard.groupby('Original feature name')['Score - Final'].min().sum())
    print(df_scorecard.groupby('Original feature name')['Score - Final'].max().sum())


    X_test_woe_transformed = woe_transform.fit_transform(X_test)
    X_test_woe_transformed.insert(0, 'Intercept', 1)
    scorecard_scores = df_scorecard['Score - Final']
    scorecard_scores = scorecard_scores.values.reshape(len(scorecard_scores), 1)
    print(X_test_woe_transformed.shape)
    print(scorecard_scores.shape)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('Best Threshold: %f' % (best_thresh))

    tr = best_thresh
    y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr,1, 0)
    confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'])
    df_cutoffs = pd.DataFrame(thresholds, columns = ['thresholds'])
    df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * 
                            ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()

    def n_approved(p):
        return np.where(y_test_proba['y_hat_test_proba'] >= p, 1, 0).sum()
    df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)
    df_cutoffs['N Rejected'] = y_test_proba['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']
    df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / y_test_proba['y_hat_test_proba'].shape[0]
    df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']
    print(df_cutoffs[df_cutoffs['thresholds'].between(best_thresh, best_thresh)])

    # woe_transform1  = WoE_Binning(X12)
    # X_test_woe_transformed1 = woe_transform1.fit_transform(X12)
    # X_test_woe_transformed1.insert(0, 'Intercept', 1)
    # X_test_woe_transformed1.fillna(value = 0, inplace = True)

    with open(r'C:\Users\vishal.lote\Documents\My Received Files\PY-SAURABH DUBEY (LAPTOP)\credit_score\TDataset\loanscore.pkl', 'wb') as file:
        pickle.dump(scorecard_scores, file)
        
    # credit_score = X_test_woe_transformed1.dot(scorecard_scores)#
    # print('Customer average Credit score: ' ,round(credit_score.mean().max()))
    # credit_score.columns = ['CREDIT_SCORE']
    # credit_score=credit_score.reset_index(drop=True)
    reg.fit(X_train, y_train)
    pickle.dump(reg, open(r'C:\Users\vishal.lote\Documents\My Received Files\PY-SAURABH DUBEY (LAPTOP)\credit_score\TDataset\npamodel.pkl','wb'))
    # npa_log = pd.DataFrame(reg.predict(X1), columns = ['NPA_LOG'])
    # npa_log =npa_log.replace({'NPA_LOG': {0: 'YES', 1: 'NO'}})
    # df2 = pd.concat([df2,credit_score,npa_log],axis = 1)
    # df2['CREDIT_SCORE']=round(df2['CREDIT_SCORE'])
    # df2
    return  res
