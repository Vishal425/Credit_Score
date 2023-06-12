#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# timestart = datetime.datetime.now()
# con = cx_Oracle.connect(db_config.user, db_config.pw, db_config.dsn)
# # print("Database version:", con.version)
# dfs_query='select * from (select * from (select py_customer_id, customer_id, customer_bank_name, customer_bank_short_name, customer_type, customer_title, customer_name, customer_creation_date, status_code as status_code_mst , customer_status_date, cust_create_from, date_of_birth, customer_age, customer_gender, cust_birth_place, martial_status, education_id, education_desc, const_code, const_desc, occupation_code, occupation_desc, community_code, community_desc, caste, caste_code, nationality_code, nationality_desc, domicile_code, domicile_desc, mobile_no, email_id, aadhaar_card, blood_group, risk_type_id, risk_type_desc, member_flag, pan, tan, form_60_flag, credit_card_flag, debit_card_flag, sms_banking_flag, mobile_banking_flag, net_banking_flag, occupation_details, employed_with, annual_income, guarantee_count, sureity_count, sureity_amount, dmat_accno, nre_nro_nri_flag, permanent_return_date, identification_id, identification_details, issue_date, issued_by, address_proof_id, address_proof_details, documents_given, kyc_complete_flag, kyc_complete_date, kyc_number, establish_licence_flag, establish_licence_no, establish_licence_date, establish_licence_exp, gst_no, gst_reg_date, gst_verify_flag, gst_verify_date, office_mst_id, old_primary_key, permanent_add, official_add, resisdential_add from py_customer_mast) m inner join py_cust_general g on m.py_customer_id=g.py_customer_id inner join py_customer_all_loans l on m.py_customer_id=l.py_customer_id)'#' where PY_CUSTOMER_ID = {}'.format(py_customer_id)

# dfs=pd.read_sql_query(dfs_query,con)
# df = dfs.loc[:,~dfs.columns.duplicated()]
# con.close()
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 500)

def Report_Data(a,connections):
    df2 = a.copy()
    set1 = {'M':'Male','F':'Female','N':'Not Specified'}
    set2 = {'M':'Married', 'N':'Not Specified', 'U':'UnMarried', 'W':'Widow', 'D':'Divorce'}
    c = tuple(a['CUSTOMER_TYPE'].unique())
    cust_id= '(%s)' % ', '.join(map(repr, c))
    dfs_query1='select * from code_master where code_mst_id in {}'.format(cust_id)
    dfs1=pd.read_sql_query(dfs_query1,connections)
    connections.close()
    set3 = dict(zip(dfs1['CODE_MST_ID'],dfs1['CODE_DESC']))
    set4 = {'N':'NOT APPLI', 'S':'SHARE HOLDER', 'L':'NOMINAL MEMBER','C':'Closed', 'A':'Active', 'H':'Hold', 'D':'Defreezed',
             'T':'Amountwise freezed', 'F':'Freezed'}
    set5 = {'Not Specified':'Not Specified','N':'NO', 'Y':'YES','A':'Additional','M':'Manually'}
    for x in a:
        if x =='CUSTOMER_GENDER':
            a[x]=a[x].map(set1)
        elif x =='MARTIAL_STATUS':
            a[x]=a[x].map(set2)
        elif x =='CUSTOMER_TYPE':
            a[x]=a[x].map(set3)
        elif x =='MODE_OF_OPERATION':
            a[x]=a[x].map(set3) 
        elif x =='MEMBER_FLAG':
            a[x]=a[x].map(set4)            
        elif x =='STATUS_CODE':
            a[x]=a[x].map(set4)
        else:
            a[x]=a[x].map(set5)
    a = a.dropna(axis=1,how='all')
    ss = a.columns
    for i in ss:
        df2[i]=a[i]
#     df2 = df2.replace(to_replace=r'\d+-',value='', regex=True).replace({np.nan: 'Not Specified'})
    return df2
# df2 = Report_Data(df,con)



# df= df.drop(['CUSTOMER_ID','CUSTOMER_BANK_NAME','CUSTOMER_CREATION_DATE','CUSTOMER_TITLE','CUSTOMER_NAME','STATUS_CODE_MST',
#              'CUSTOMER_STATUS_DATE','DATE_OF_BIRTH','CUST_BIRTH_PLACE','EDUCATION_DESC','CONST_DESC',
#         'OCCUPATION_DESC','COMMUNITY_DESC','CASTE','NATIONALITY_DESC','DOMICILE_DESC','EMAIL_ID','RISK_TYPE_DESC','TAN',
#              'OCCUPATION_DETAILS','EMPLOYED_WITH','DMAT_ACCNO','PERMANENT_RETURN_DATE','IDENTIFICATION_DETAILS','ISSUE_DATE',
#         'ISSUED_BY','ADDRESS_PROOF_DETAILS','DOCUMENTS_GIVEN','KYC_COMPLETE_DATE','KYC_NUMBER','ESTABLISH_LICENCE_NO',
#              'ESTABLISH_LICENCE_DATE','ESTABLISH_LICENCE_EXP','GST_NO','GST_REG_DATE','GST_VERIFY_DATE','OFFICE_MST_ID',
#             'OLD_PRIMARY_KEY','OFFICIAL_ADD','RESISDENTIAL_ADD','PERMANENT_ADD','PY_CUST_GEN_MST_ID',
#         'FOREGIN_TOUR_YN','CUST_FRDT_IN_MTHS','SUREITY_TO_NOOFAC', 'PY_CUST_MST_ID_ALL_LOANS','PY_CUST_MST_ID_ALL',
#         'PY_CUSTCCOD_MST_ID','PY_CUSTLN_MST_ID','GLTITLE','NEW_AC_NUMBER','ACCOUNT_NAME','ACCOUNT_OPEN_DATE',
#         'ACCOUNT_STATUS_DATE','DP_AMOUNT','EXPIRY_DATE','LAST_INTT_RECEIVED_DT','LIEN_DATE','NPA_DATE',
#             'SANCTION_DATE','SECURITY_CODE','VALUE_DESCRIPTION','LOAN_TYPE','BLOOD_GROUP','GLCODE'],axis = 1)

# df_cat = df.select_dtypes(include = 'object').copy()
# df_num = df.select_dtypes(include = 'number').copy()
# columns_list = list(df_cat.columns)
def Mobile_adhar_pan_flag(df1):
    df_1 = []
    for columns in list(df1.columns):
        df_1.append(df1[columns].apply(lambda x: 'YES' if len(str(x)) == 10 or len(str(x)) == 12  else 'NO'))  
    df_1 = pd.concat(df_1, axis = 1)
    sd=list(df_1.columns[(df_1 == 'YES').any(axis=0)])
    df1[sd]=df_1[sd]
    return df1
# df_cat = Mobile_adhar_pan_flag(df_cat)

def Missing_Treatment(dfn,dfc):
    df_1 = []
    dfn.fillna(0, inplace=True)
    dfc.fillna(dfc.mode().iloc[0], inplace=True)
    cols = list(dfn.columns)
    for col in cols:
        dfn[col] = pd.to_numeric(dfn[col].map(lambda x: str(x).lstrip('-').rstrip('-')))
    df1 = pd.concat([dfn, dfc], axis = 1)    
    return df1
# df = Missing_Treatment(df_num,df_cat)

# df['good_bad'] = np.where(df.loc[:, 'NPA_CLASS_ID'].isin(['ST','SS']), 1, 0)
# # Drop the original column
# df = df.drop(columns = ['NPA_CLASS_ID'])
# X = df.drop('good_bad', axis = 1)
# y = df['good_bad']

# train= df.loc[df['STATUS_CODE'] == 'C']
# X_train = train.drop('good_bad', axis = 1)
# y_train = train['good_bad']

# test= df.loc[df['STATUS_CODE'] != 'C']
# X_test = test.drop('good_bad', axis = 1)
# y_test = test['good_bad']

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def Chi_square(dfs,ytrain):
    chi2_check = {}
    # loop over each column in the training set to calculate chi-statistic with the categorical target variable
    for column in dfs:
        chi, p, dof, ex = chi2_contingency(pd.crosstab(ytrain, dfs[column]))
        chi2_check.setdefault('Feature',[]).append(column)
        chi2_check.setdefault('p-value',[]).append(round(p, 10))
    # convert the dictionary to a DF
    chi2_result = pd.DataFrame(data = chi2_check)
    chi2_result.sort_values(by = ['p-value'], ascending = True, inplace = True)
    chi = chi2_result.reset_index(drop=True)
    chi = chi[chi['p-value']== 000000e+00]
    chi2=list(chi['Feature'])
    return chi2
# df_cat = df.select_dtypes(include = 'object').copy()
# ch = Chi_square(df_cat,y_train)


def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df
# apply to our final  categorical variables
# X_train = dummy_creation(X_train, ch)
# X_test = dummy_creation(X_test, ch)
# X2 = df.drop('good_bad', axis = 1)
# X_dummy = dummy_creation(X2, ch)
# X4 = dummy_creation(X, ch)
# # reindex the dummied test set variables to make sure all the feature columns in the train set are also available in the test set
# X3 = X4.reindex(labels=X4.columns, axis=1, fill_value=0)
# X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)

def Anova(dfs,dependent):
    F_statistic, p_values = f_classif(dfs, dependent)
    # convert to a DF
    ANOVA_F_table = pd.DataFrame(data = {'Numerical_Feature': dfs.columns.values, 'F-Score': F_statistic, 'p values':        p_values.round(decimals=10)})
    ANOVA_F_table.sort_values(by = ['F-Score'], ascending = False, inplace = True)
    ANOVA_F_table = ANOVA_F_table.reset_index(drop=True)
    anova = ANOVA_F_table[ANOVA_F_table['p values']==000000e+00]
    anova1=list(anova['Numerical_Feature'])
    return anova1
# anova = Anova(df_num,y)
 
def woe_ordered_continuous(df, continuous_variabe_name, y_df):
    df = pd.concat([df[continuous_variabe_name], y_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

# # Create copies of the 4 training sets to be preprocessed using WoE
# X_train_prepr = X_train.copy()
# y_train_prepr = y_train.copy()

def Numerical_Features_Range(dfs,dfv,anova_feature):
    df_temp3 = []
    for i in range (len(anova_feature)):
        np.seterr(divide = 'ignore')
        dfs[anova_feature[i]] = pd.cut(dfs[anova_feature[i]], 25)
        df_temp = woe_ordered_continuous(dfs, anova_feature[i], dfv)
        df_temp = df_temp.replace([np.inf, -np.inf], np.nan)
        df_temp=df_temp[df_temp.WoE.notnull()].reset_index()
        df_temp=pd.DataFrame(df_temp[anova_feature[i]])
        df_temp[anova_feature[i]] = df_temp[anova_feature[i]].map(lambda x: str(x).lstrip('(').rstrip(']'))
        df_temp2=pd.get_dummies(df_temp,columns =[anova_feature[i]], prefix = anova_feature[i], prefix_sep = ':')
        ss=list(df_temp2.columns[(df_temp2 == 1).any(axis=0) ])
        df_temp3.append(df_temp2[ss].columns)
    return df_temp3
# df_temp3 = Numerical_Features_Range(X_train_prepr,y_train_prepr,anova)



def Training_woe_transform(dfs,woe,pipe):
    X_train_woe_transformed = woe.fit_transform(dfs)
    feature_name = X_train_woe_transformed.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(pipe['model'].coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', pipe['model'].intercept_[0]]
    summary_table.sort_index(inplace = True)
    return summary_table
# summary_table = Training_woe_transform(X_train,woe_transform,pipeline)


    # make preditions on our test set
def prediction_testing_set(dfx,dfy,pipe):
    y_hat_test = pipe.predict(dfx)
    # get the predicted probabilities
    y_hat_test_proba = pipe.predict_proba(dfx)
    # select the probabilities of only the positive class (class 1 - default) 
    y_hat_test_proba = y_hat_test_proba[:][: , 1]
    # we will now create a new DF with actual classes and the predicted probabilities
    # create a temp y_test DF to reset its index to allow proper concaternation with y_hat_test_proba
    y_test_temp = dfy.copy()
    y_test_temp.reset_index(drop = True, inplace = True)
    y_test_proba = pd.concat([y_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
    # check the shape to make sure the number of rows is same as that in y_test
    y_test_proba.shape
    # Rename the columns
    y_test_proba.columns = ['y_test_class_actual', 'y_hat_test_proba']
    # Makes the index of one dataframe equal to the index of another dataframe.
    y_test_proba.index = dfx.index
    return y_test_proba
# y_test_proba = prediction_testing_set(X_test,y_test,pipeline)

# In[ ]:




