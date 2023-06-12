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
# import db_config
import datetime
import itertools
import warnings
from score.db_function import *

pd.options.mode.chained_assignment = None  # default='warn'
from django.db import connection 

timestart = datetime.datetime.now()
# con = cx_Oracle.connect(db_config.username, db_config.password, db_config.dsn)
# py_customer_id=int(input('Enter the customer id :'))
def sql_value(val1):

    py_customer_id = val1
    dfs_query='select * from (select * from (select py_customer_id, customer_id, customer_bank_name, customer_bank_short_name, customer_type, customer_title, customer_name, customer_creation_date, status_code as status_code_mst , customer_status_date, cust_create_from, date_of_birth, customer_age, customer_gender, cust_birth_place, martial_status, education_id, education_desc, const_code, const_desc, occupation_code, occupation_desc, community_code, community_desc, caste, caste_code, nationality_code, nationality_desc, domicile_code, domicile_desc, mobile_no, email_id, aadhaar_card, blood_group, risk_type_id, risk_type_desc, member_flag, pan, tan, form_60_flag, credit_card_flag, debit_card_flag, sms_banking_flag, mobile_banking_flag, net_banking_flag, occupation_details, employed_with, annual_income, guarantee_count, sureity_count, sureity_amount, dmat_accno, nre_nro_nri_flag, permanent_return_date, identification_id, identification_details, issue_date, issued_by, address_proof_id, address_proof_details, documents_given, kyc_complete_flag, kyc_complete_date, kyc_number, establish_licence_flag, establish_licence_no, establish_licence_date, establish_licence_exp, gst_no, gst_reg_date, gst_verify_flag, gst_verify_date, office_mst_id, old_primary_key, permanent_add, official_add, resisdential_add from py_customer_mast) m inner join py_cust_general g on m.py_customer_id=g.py_customer_id inner join py_customer_all_loans l on m.py_customer_id=l.py_customer_id)  where PY_CUSTOMER_ID = {}'.format(py_customer_id)
    df=pd.read_sql_query(dfs_query,connection)
    df = df.loc[:,~df.columns.duplicated()]
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_rows', 500)

    #from db_function import Report_Data
    df2 = Report_Data(df,connection)

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

    #from db_function import Mobile_adhar_pan_flag
    df_cat = df.select_dtypes(include = 'object').copy()
    df_num = df.select_dtypes(include = 'number').copy()
    df_cat = Mobile_adhar_pan_flag(df_cat)

    #from db_function import Missing_Treatment
    df = Missing_Treatment(df_num,df_cat)

    # df.fillna(value = 0, inplace = True)

    df['good_bad'] = np.where(df.loc[:, 'NPA_CLASS_ID'].isin(['ST','SS']), 1, 0)
    # Drop the original column
    df = df.drop(columns = ['NPA_CLASS_ID'])
    X = df.drop('good_bad', axis = 1)
    y = df['good_bad']

    #from db_function import MultiColumnLabelEncoder
    df_cat = X.select_dtypes(include = 'object').copy()    
    cat_cols = list(df_cat.columns)
    X4 = MultiColumnLabelEncoder(columns = cat_cols).fit_transform(X)
    X1 = X4.reindex(labels=X4.columns, axis=1, fill_value=0)

    # df2 = df2.loc[df2['PY_CUSTOMER_ID'] == py_customer_id]
    # df2 = df2.reset_index(drop=True)
    # X1 = X1.loc[X1['PY_CUSTOMER_ID'] == py_customer_id]

    ds = pd.read_csv(r"C:\Users\vishal.lote\Documents\My Received Files\PY-SAURABH DUBEY (LAPTOP)\credit_score\TDataset\file.csv")
    ds = ds.iloc[:,1:]
    cols = list(ds['0'].values)
    cols2 = pd.DataFrame(cols)
    cols1 = cols2[0].str.split(':').str[0]        
    mylist = cols2[0].str.split(':').str[1]
    cols_list = list(X.columns)

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

    reg = LogisticRegression(max_iter=500, class_weight = 'balanced')
    woe_transform1  = WoE_Binning(X1)
    X_test_woe_transformed1 = woe_transform1.fit_transform(X1)
    X_test_woe_transformed1.insert(0, 'Intercept', 1)
    X_test_woe_transformed1.fillna(value = 0, inplace = True)

    pipline_from_pickle = pickle.load(open(r'C:\Users\vishal.lote\Documents\My Received Files\PY-SAURABH DUBEY (LAPTOP)\credit_score\TDataset\loanscore.pkl','rb'))
    credit_score = X_test_woe_transformed1.dot(pipline_from_pickle)
    credit_score.columns = ['CREDIT_SCORE']
    credit_score=credit_score.reset_index(drop=True)
    model = pickle.load(open(r'C:\Users\vishal.lote\Documents\My Received Files\PY-SAURABH DUBEY (LAPTOP)\credit_score\TDataset\npamodel.pkl','rb'))
    X1.fillna(value = '0',inplace = True)
    npa_log = pd.DataFrame(model.predict(X1), columns = ['NPA_LOG'])
    npa_log =npa_log.replace({'NPA_LOG': {0: 'YES', 1: 'NO'}})
    df2 = df2.replace(to_replace=r'\d+-',value='', regex=True).replace({np.nan: 'Not Specified'})    
    df2 = pd.concat([df2,credit_score,npa_log],axis = 1)

    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds(), 2)
    print ("Time taken to execute above cell: " + str(timedelta) + " seconds")    

    return df2