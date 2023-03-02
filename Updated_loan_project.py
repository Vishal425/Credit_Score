import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import cx_Oracle
import pickle
import db_config
import datetime
timestart = datetime.datetime.now()

con = cx_Oracle.connect(db_config.user, db_config.pw, db_config.dsn)
# print("Database version:", con.version)
dfs_query='select * from (select * from (select py_customer_id, customer_id, customer_bank_name, customer_bank_short_name, customer_type, customer_title, customer_name, customer_creation_date, status_code as status_code_mst , customer_status_date, cust_create_from, date_of_birth, customer_age, customer_gender, cust_birth_place, martial_status, education_id, education_desc, const_code, const_desc, occupation_code, occupation_desc, community_code, community_desc, caste, caste_code, nationality_code, nationality_desc, domicile_code, domicile_desc, mobile_no, email_id, aadhaar_card, blood_group, risk_type_id, risk_type_desc, member_flag, pan, tan, form_60_flag, credit_card_flag, debit_card_flag, sms_banking_flag, mobile_banking_flag, net_banking_flag, occupation_details, employed_with, annual_income, guarantee_count, sureity_count, sureity_amount, dmat_accno, nre_nro_nri_flag, permanent_return_date, identification_id, identification_details, issue_date, issued_by, address_proof_id, address_proof_details, documents_given, kyc_complete_flag, kyc_complete_date, kyc_number, establish_licence_flag, establish_licence_no, establish_licence_date, establish_licence_exp, gst_no, gst_reg_date, gst_verify_flag, gst_verify_date, office_mst_id, old_primary_key, permanent_add, official_add, resisdential_add from py_customer_mast) m inner join py_cust_general g on m.py_customer_id=g.py_customer_id inner join py_customer_all_loans l on m.py_customer_id=l.py_customer_id)'#' where PY_CUSTOMER_ID = {}'.format(py_customer_id)
dfs=pd.read_sql_query(dfs_query,con)
df = dfs.loc[:,~dfs.columns.duplicated()]
con.close()

df2 = df.copy()


set={'1-Ssc':'SSC', '0-N.A.':'NA' ,'3-Graduation':'GRADUATION' ,'2-Hsc':'HSC' ,'4-Postgraduation':'POSTGRADUATION',
     'N':'NO', 'Y':'YES','1-Individual':'Individual',
      '2-Proprietary Firm':'Proprietary Firm', '6-Coop Society':'Coop Society', '8-Bank':'Bank',
       '7-Trust':'Trust', '3-Partnership Firm':'Partnership Firm', '10-HUF':'HUF','2-Business':'Business', 
      '7-Other':'Other', '3-House Wife':'House Wife', '4-Professional':'Professional',
       '10-AGRICULTURE / FARMER':'FARMER', '1-Service':'Service', '5-Retired':'Retired', '11-DOCTOR':'DOCTOR',
       '9-PRIVATE EMPLOYEE':'PRIVATE EMPLOYEE', '12-ADVOCATE':'ADVOCATE', '6-Student':'Student',
       '8-GOVT. EMPLOYEE':'GOVT EMPLOYEE', '17-PENSHIONER':'PENSHIONER',
       '13-ENGINEER':'ENGINEER', '19-UNEMPLOYED':'UNEMPLOYED',  '14-TRADING':'TRADING',
       '18-EX-SERVICE MAN':'EXSERVICE MAN', '16-COMMISSION AGENT':'COMMISSION AGENT','1-Hindu':'Hindu', '0-None':'None', '3-Muslim':'Muslim', 
      '2-Sikh':'Sikh', '6-Buddhists':'Buddhists','4-Christian':'Christian','1-Albanians':'Albanians', '82-Indians':'Indians',
      '77-Guyanese':'Guyanese', '59-Faroese':'Faroese',
       '87-Israelis':'Israelis', '174-Vanuatuans':'Vanuatuans', '111-Manx':'Manx','1-Indian':'Indian', '2-Others':'Others',
       '3-High Risk':'High Risk', '1-Low Risk':'Low Risk', '2-Medium Risk':'Medium Risk','M':'Manualy'}
set1 = {'M':'Male','F':'Female','N':'Not Specified'}
set2 = {'M':'Married', 'N':'Not Specified', 'U':'UnMarried', 'W':'Widow', 'D':'Divorce'}
set3 = {166:'General Customer',168:'Firm', 173:'Senior Citizen', 
      174:'Society',175:'Staff',177:'Student',178:'Trust',
        179:'Women',574:'Handicap',1414:'NA',181:'Self',182:'Proprietor', 
      183:'Either Or Survivor', 184:'Former Or Survivor',185:'Jointly Or Severally',186:'Anyone Single Or Survivor',
      187:'Any Two Jointly',189:'All Jointly',190:'Only First',192:'No 1,2,3 Jointly',197:'Authorised Signatory',
      199:'Managing Director',207:'Minor And Natural Guardian',208:'Minor And Legal Guardian',209:'Minor Alone',
      912:'NA',932:'NA'}
set4 = {'N':'NOT APPLI', 'S':'SHARE HOLDER', 'L':'NOMINAL MEMBER','C':'Closed', 'A':'Active', 'H':'Hold', 'D':'Defreezed',
         'T':'Amountwise freezed', 'F':'Freezed'}
df2_cat = df2.copy()
for x in df2_cat:
    if x =='CUSTOMER_GENDER':
        df2_cat[x]=df2_cat[x].map(set1)
    elif x =='MARTIAL_STATUS':
        df2_cat[x]=df2_cat[x].map(set2)
    elif x =='CUSTOMER_TYPE':
        df2_cat[x]=df2_cat[x].map(set3)
    elif x =='MODE_OF_OPERATION':
        df2_cat[x]=df2_cat[x].map(set3) 
    elif x =='MEMBER_FLAG':
        df2_cat[x]=df2_cat[x].map(set4)            
    elif x =='STATUS_CODE':
        df2_cat[x]=df2_cat[x].map(set4) 
    else:    
        df2_cat[x] = df2_cat[x].map(set)
df2_cat = df2_cat.dropna(axis=1,how='all')
ss = df2_cat.columns
df2[ss]=df2_cat[ss]

df= df.drop(['CUSTOMER_ID','CUSTOMER_BANK_NAME','CUSTOMER_CREATION_DATE','CUSTOMER_TITLE','CUSTOMER_NAME','STATUS_CODE_MST',
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

df_cat1 = df.select_dtypes(include = 'object').copy()
cols = list(df_cat1.columns)
for columns in cols:
    df_cat1[columns] = df_cat1[columns].apply(lambda x: 'YES' if len(str(x)) == 10 or len(str(x)) == 12  else 'NO')
sd=list(df_cat1.columns[(df_cat1 == 'YES').any(axis=0)])
df[sd]=df_cat1[sd]

df_cat = df.select_dtypes(include = 'object').copy()
df_num = df.select_dtypes(include = 'number').copy()
df_num.fillna(0, inplace=True)
df_cat.fillna(df_cat.mode().iloc[0], inplace=True)
cols = list(df_num.columns)
for col in cols:
    df_num[col] = pd.to_numeric(df_num[col].map(lambda x: str(x).lstrip('-').rstrip('-')))
df = pd.concat([df_num,df_cat],axis = 1)

df['good_bad'] = np.where(df.loc[:, 'NPA_CLASS_ID'].isin(['ST','SS']), 1, 0)
# Drop the original column
df.drop(columns = ['NPA_CLASS_ID'], inplace = True)
X = df.drop('good_bad', axis = 1)
y = df['good_bad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100, stratify = y)

#Lebal Encoding for NPA generations
from sklearn.preprocessing import LabelEncoder
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
df_cat = df.select_dtypes(include = 'object').copy()    
cols = list(df_cat.columns)
X1=MultiColumnLabelEncoder(columns = cols).fit_transform(X)
X_train1=MultiColumnLabelEncoder(columns = cols).fit_transform(X_train)
X_test1=MultiColumnLabelEncoder(columns = cols).fit_transform(X_test)

chi2_check = {}
# loop over each column in the training set to calculate chi-statistic with the categorical target variable
for column in df_cat:
    chi, p, dof, ex = chi2_contingency(pd.crosstab(y_train, df_cat[column]))
    chi2_check.setdefault('Feature',[]).append(column)
    chi2_check.setdefault('p-value',[]).append(round(p, 10))
# convert the dictionary to a DF
chi2_result = pd.DataFrame(data = chi2_check)
chi2_result.sort_values(by = ['p-value'], ascending = True, inplace = True)
chi = chi2_result.reset_index(drop=True)
chi = chi[chi['p-value']== 000000e+00]
chi = list(chi['Feature'])

# function to create dummy variables
def dummy_creation(df, columns_list):
    df_dummies = []
    for col in columns_list:
        df_dummies.append(pd.get_dummies(df[col], prefix = col, prefix_sep = ':'))
    df_dummies = pd.concat(df_dummies, axis = 1)
    df = pd.concat([df, df_dummies], axis = 1)
    return df
# apply to our final  categorical variables
X_train = dummy_creation(X_train, chi)

X_test = dummy_creation(X_test, chi)

X2 = df.drop('good_bad', axis = 1)
X_dummy = dummy_creation(X2, chi)
X4 = dummy_creation(X, chi)
# reindex the dummied test set variables to make sure all the feature columns in the train set are also available in the test set
X3 = X4.reindex(labels=X4.columns, axis=1, fill_value=0)
X_test = X_test.reindex(labels=X_train.columns, axis=1, fill_value=0)

py_customer_id=int(input('Enter the customer id: '))
X1 = X1.loc[X1['PY_CUSTOMER_ID'] == py_customer_id]
X12 = X3.copy()
X12 = X12.loc[X12['PY_CUSTOMER_ID'] == py_customer_id]

# Create copies of the 4 training sets to be preprocessed using WoE
X_train_prepr = X_train.copy()
y_train_prepr = y_train.copy()
X_test_prepr = X_test.copy()
y_test_prepr = y_test.copy()

# Calculate F Statistic and corresponding p values
F_statistic, p_values = f_classif(df_num, y)
# convert to a DF
ANOVA_F_table = pd.DataFrame(data = {'Numerical_Feature': df_num.columns.values, 'F-Score': F_statistic, 'p values': p_values.round(decimals=10)})
ANOVA_F_table.sort_values(by = ['F-Score'], ascending = False, inplace = True)
ANOVA_F_table = ANOVA_F_table.reset_index(drop=True)
anova = ANOVA_F_table[ANOVA_F_table['p values']==000000e+00]
anova = list(anova['Numerical_Feature'])

# We define a function to calculate WoE of continuous variables. This is same as the function we defined earlier for discrete variables.
# The only difference are the 2 commented lines of code in the function that results in the df being sorted by continuous variable values
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

df_temps = []
df_temp3 = []
for i in range (len(anova)):
    X_train_prepr[anova[i]] = pd.cut(X_train_prepr[anova[i]], 25)
    df_temp = woe_ordered_continuous(X_train_prepr, anova[i], y_train_prepr)
    df_temp = df_temp.replace([np.inf, -np.inf], np.nan)
    df_temp=df_temp[df_temp.WoE.notnull()].reset_index()
    df_temp=pd.DataFrame(df_temp[anova[i]])
    df_temp[anova[i]] = df_temp[anova[i]].map(lambda x: str(x).lstrip('(').rstrip(']'))
    df_temps.append(df_temp)
    df_temp2=pd.get_dummies(df_temp,columns =[anova[i]], prefix = anova[i], prefix_sep = ':')
    ss=list(df_temp2.columns[(df_temp2 == 1).any(axis=0) ])
    df_temp3.append(df_temp2[ss].columns)

import itertools
my_list = list(itertools.chain(*df_temp3))

cols = my_list
cols2 = pd.DataFrame(cols)
cols1 = cols2[0].str.split(':').str[0]        
mylist = cols2[0].str.split(':').str[1]

cols_1 = list(X_dummy.drop(X_dummy.iloc[:, 0:len(X2.columns)].columns, axis = 1) )

class WoE_Binning(BaseEstimator, TransformerMixin):
    def __init__(self, X): # no *args or *kargs
        self.X = X
    def fit(self, X, y = None):
        return self #nothing else to do
    def transform(self, X):
        X_new = X.loc[:,cols_1]
        for i in range (len(mylist)):
            X_new[cols[i]] = np.where((X[cols1[i]] > float(mylist[i].split(',')[0])) & (X[cols1[i]] <= float(mylist[i].split(',')[1])), 1, 0)
        return X_new

reg = LogisticRegression(max_iter=1000, class_weight = 'balanced')
woe_transform = WoE_Binning(X)
pipeline = Pipeline(steps=[('woe', woe_transform), ('model', reg)])
pipeline.fit(X_train, y_train)

# first create a transformed training set through our WoE_Binning custom class
X_train_woe_transformed = woe_transform.fit_transform(X_train)
# Store the column names in X_train as a list
feature_name = X_train_woe_transformed.columns.values
# Create a summary table of our logistic regression model
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
# Create a new column in the dataframe, called 'Coefficients', with row values the transposed coefficients from the 'LogisticRegression' model
summary_table['Coefficients'] = np.transpose(pipeline['model'].coef_)
# Increase the index of every row of the dataframe with 1 to store our model intercept in 1st row
summary_table.index = summary_table.index + 1
# Assign our model intercept to this new row
summary_table.loc[0] = ['Intercept', pipeline['model'].intercept_[0]]
# Sort the dataframe by index
summary_table.sort_index(inplace = True) 

# make preditions on our test set
y_hat_test = pipeline.predict(X_test)
# get the predicted probabilities
y_hat_test_proba = pipeline.predict_proba(X_test)
# select the probabilities of only the positive class (class 1 - default) 
y_hat_test_proba = y_hat_test_proba[:][: , 1]
# we will now create a new DF with actual classes and the predicted probabilities
# create a temp y_test DF to reset its index to allow proper concaternation with y_hat_test_proba
y_test_temp = y_test.copy()
y_test_temp.reset_index(drop = True, inplace = True)
y_test_proba = pd.concat([y_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
# check the shape to make sure the number of rows is same as that in y_test
y_test_proba.shape
# Rename the columns
y_test_proba.columns = ['y_test_class_actual', 'y_hat_test_proba']
# Makes the index of one dataframe equal to the index of another dataframe.
y_test_proba.index = X_test.index

# assign a threshold value to differentiate good with bad
tr = 0.5
# crate a new column for the predicted class based on predicted probabilities and threshold
# We will determine this optimat threshold later in this project
y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr, 1, 0)
# create the confusion matrix
confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'])
# get the values required to plot a ROC curve
fpr, tpr, thresholds = roc_curve(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
# Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) on our test set
AUROC = roc_auc_score(y_test_proba['y_test_class_actual'], y_test_proba['y_hat_test_proba'])
# calculate Gini from AUROC
Gini = AUROC * 2 - 1

# Concatenates two dataframes.
df_scorecard = pd.concat([summary_table])
# We reset the index of a dataframe.
df_scorecard.reset_index(inplace = True)
# create a new column, called 'Original feature name', which contains the value of the 'Feature name' column, up to the column symbol.
df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

# Define the min and max threshholds for our scorecard
min_score = 300
max_score = 850
# calculate the sum of the minimum coefficients of each category within the original feature name
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
# calculate the sum of the maximum coefficients of each category within the original feature name
max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
# create a new columns that has the imputed calculated Score based on the multiplication of the coefficient by the ratio of the differences between
# maximum & minimum score and maximum & minimum sum of cefficients.
df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
# update the calculated score of the Intercept (i.e. the default score for each loan)
df_scorecard.loc[0, 'Score - Calculation'] = ((df_scorecard.loc[0,'Coefficients'] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
# round the values of the 'Score - Calculation' column and store them in a new column
df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
print(df_scorecard['Score - Preliminary'])
# check the min and max possible scores of our scorecard
min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()

df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']

#First create a transformed test set through our WoE_Binning custom class and insert an Intercept column
# look like we can get by deducting 1 from the Intercept
df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
# df_scorecard.loc[0, 'Score - Final'] = 568
# Recheck min and max possible scores
print(df_scorecard.groupby('Original feature name')['Score - Final'].min().sum())
print(df_scorecard.groupby('Original feature name')['Score - Final'].max().sum())

# first create a transformed test set through our WoE_Binning custom class
X_test_woe_transformed = woe_transform.fit_transform(X_test)
# insert an Intercept column in its beginning to align with the # of rows in scorecard
X_test_woe_transformed.insert(0, 'Intercept', 1)
X_test_woe_transformed.head()
# get the list of our final scorecard scores
scorecard_scores = df_scorecard['Score - Final']
# check the shapes of test set and scorecard before doing matrix dot multiplication
# print(X_test_woe_transformed.shape)
# print(scorecard_scores.shape)
scorecard_scores = scorecard_scores.values.reshape(len(scorecard_scores), 1)
print(X_test_woe_transformed.shape)
print(scorecard_scores.shape)

#Setting loan approval cut-offs
# Calculate Youden's J-Statistic to identify the best threshhold
J = tpr - fpr
# locate the index of the largest J
ix = np.argmax(J)
best_thresh = thresholds[ix]
print('Best Threshold: %f' % (best_thresh))

# update the threshold value
tr = best_thresh
# crate a new column for the predicted class based on predicted probabilities and threshold
y_test_proba['y_test_class_predicted'] = np.where(y_test_proba['y_hat_test_proba'] > tr,1, 0)
# create the confusion matrix
confusion_matrix(y_test_proba['y_test_class_actual'], y_test_proba['y_test_class_predicted'])
# create a new DF comprising of the thresholds from the ROC output
df_cutoffs = pd.DataFrame(thresholds, columns = ['thresholds'])
# calcue Score corresponding to each threshold
df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * 
                       ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()

def n_approved(p):
    return np.where(y_test_proba['y_hat_test_proba'] >= p, 1, 0).sum()

# Assuming that all credit applications above a given probability of being 'good' will be approved,
# when we apply the 'n_approved' function to a threshold, it will return the number of approved applications.
# Thus, here we calculate the number of approved appliations for all thresholds.
df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)
# Then, we calculate the number of rejected applications for each threshold.
# It is the difference between the total number of applications and the approved applications for that threshold.
df_cutoffs['N Rejected'] = y_test_proba['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']
# Approval rate equalts the ratio of the approved applications and all applications.
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / y_test_proba['y_hat_test_proba'].shape[0]
# Rejection rate equals one minus approval rate.
df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']

# let's have a look at the approval and rejection rates at our ideal threshold
print(df_cutoffs[df_cutoffs['thresholds'].between(best_thresh, best_thresh)])


#  WoE_Binning custom class on individual customer data
woe_transform1  = WoE_Binning(X12)
# first create a transformed test set through our WoE_Binning custom class
X_test_woe_transformed1 = woe_transform1.fit_transform(X12)
# insert an Intercept column in its beginning to align with the # of rows in scorecard
X_test_woe_transformed1.insert(0, 'Intercept', 1)
X_test_woe_transformed1.fillna(value = 0, inplace = True)

credit_score = X_test_woe_transformed1.dot(scorecard_scores)
print('Customer average Credit score: ' ,round(credit_score.mean().max()))
reg.fit(X_train1, y_train)
array = reg.predict(X1)
for i in array:
    if (i == 1):
        print ("NPA: NO")
    else:
        print ("NPA: YES")
print('Customer Credit score: ' ,round(credit_score))


# PRINT HOW MUCH TIME IT TOOK TO RUN THE CELL
timeend = datetime.datetime.now()
timedelta = round((timeend-timestart).total_seconds(), 2)
print ("Time taken to execute above cell: " + str(timedelta) + " seconds")    