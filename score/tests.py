# from django.test import TestCase
# from . import views
# from  django.http import HttpResponse
# from django.template.loader import get_template
# from .utils import render_to_pdf
# from django.views.generic import View
# from score.test_sql import *

# # Create your tests here.
# class GeneratePDF(View):
#     def get(self, request, *args, **kwargs):
#         template = get_template('test.html')
#         json_Value=(sql_value())
#         html = template.render(json_Value)
#         pdf = render_to_pdf('test.html', {'json_Value':json_Value})
#         if pdf:
#             response = HttpResponse(pdf, content_type='application/pdf')
#             filename = "Invoice_%s.pdf" %("12341231")
#             content = "inline; filename='%s'" %(filename)
#             download = request.GET.get("download")
#             if download:
#                 content = "attachment; filename='%s'" %(filename)
#             response['Content-Disposition'] = content
#             return response
#         return HttpResponse("Not found")

import pandas as pd
df2=pd.read_csv(r'C:\Users\saurabh.dubey\Desktop\Virtual_Enrvironment\python_project\credit_analysis\Dataset\combine_1000_cust.csv')
#print(df2.head(3))

df=df2[['EDUCATION_ID','EDUCATION_DESC']]
A = list(df2['EDUCATION_ID']).split('-')
B= list(df2['EDUCATION_DESC'])

zip_iterator = dict(zip(A, B))
print(zip_iterator)




# def func(CUSTOMER_GENDER):
#     if CUSTOMER_GENDER=='F':
#         return 'Female'
#     elif CUSTOMER_GENDER=='M':
#         return 'Male'
#     return 'Not Specified'
# df2['CUSTOMER_GENDER']=df2['CUSTOMER_GENDER'].apply(func)

# def func(MARTIAL_STATUS):
#     if MARTIAL_STATUS=='D':
#         return 'Divorce'
#     elif MARTIAL_STATUS=='M':
#         return 'Married'
#     elif MARTIAL_STATUS=='U':
#         return 'UnMarried'
#     elif MARTIAL_STATUS=='W':
#         return 'Widow'
#     return 'Not Specified'
# df2['MARTIAL_STATUS']=df2['MARTIAL_STATUS'].apply(func)

# def func(EDUCATION_ID):
#     if EDUCATION_ID==438:
#         return 'SSC'
#     elif EDUCATION_ID==439:
#         return 'HSC'
#     elif EDUCATION_ID==440:
#         return 'Graduation'
#     elif EDUCATION_ID==441:
#         return 'Postgraduation'
#     elif EDUCATION_ID==442:
#         return 'NA'
#     return 'No'
# df2['EDUCATION_ID']=df2['EDUCATION_ID'].apply(func)

# def func(NATIONALITY):
#     if NATIONALITY==445:
#         return 'Albanians'
#     elif NATIONALITY==1034:
#         return 'Indians'
#     elif NATIONALITY==1029:
#         return 'Guyanese'
#     elif NATIONALITY==1011:
#         return 'Faroese'
#     elif NATIONALITY==1039:
#         return 'Israelis'
#     elif NATIONALITY==1026:
#         return 'Guianese (French)'
#     elif NATIONALITY==1063:
#         return 'Manx'
#     return 'No'
# df2['NATIONALITY_CODE']=df2['NATIONALITY_CODE'].apply(func)

# def func(COMMUNITY):
#     if COMMUNITY==138:
#         return 'Hindu'
#     elif COMMUNITY==139:
#         return 'Sikh'
#     elif COMMUNITY==140:
#         return 'Muslim'
#     elif COMMUNITY==141:
#         return 'Christian'
#     elif COMMUNITY==143:
#         return 'Buddhists'
#     elif COMMUNITY==144:
#         return 'None'
#     elif COMMUNITY==0:
#         return 'None'
#     return 'No'
# df2['COMMUNITY_CODE']=df2['COMMUNITY_CODE'].apply(func)

# occup = {131:'Service',132:'Business', 133:'House Wife', 134:'Professional',135:'Retired',136:'Student',137:'Other',
#         929:'GOVT EMPLOYEE',930:'PRIVATE EMPLOYEE',931:'AGRICULTURE',932:'DOCTOR',933:'ADVOCATE',934:'ENGINEER',
#         935:'TRADING',936:'NA',937:'COMMISSION AGENT',938:'PENSHIONER',939:'EXSERVICE MAN',940:'UNEMPLOYED',
#         0:'None'}

# df2["OCCUPATION_CODE"]=df2["OCCUPATION_CODE"].map(occup)


# caste = {145:'None',146:'SC', 147:'ST', 148:'OBC',149:'NT',150:'General',151:'Other',
#         0:'NA'}
# df2['CASTE_CODE']=df2['CASTE_CODE'].map(caste)


# dom = {443:'Indian',444:'Others'}
# df2["DOMICILE_CODE"]=df2["DOMICILE_CODE"].map(dom)

# cust = {166:'General Customer',168:'Firm', 173:'Senior Citizen', 174:'Society',175:'Staff',177:'Student',178:'Trust',
#         179:'Women',574:'Handicap',1414:'NA'}
# df2['CUSTOMER_TYPE']=df2['CUSTOMER_TYPE'].map(cust)

# cust1 = {'M':'Manually'}
# df2['CUST_CREATE_FROM']=df2['CUST_CREATE_FROM'].map(cust1)

# risk = {447:'Low Risk',448:'Medium Risk',681:'High Risk'}
# df2['RISK_TYPE_ID']=df2['RISK_TYPE_ID'].map(risk)

# const = {122:'Individual',123:'Proprietary Firm', 124:'Partnership Firm', 127:'Coop Society',128:'Trust',129:'Bank',606:'HUF'}
# df2['CONST_CODE']=df2['CONST_CODE'].map(const)

# mode = {181:'Self',182:'Proprietor', 183:'Either Or Survivor', 184:'Former Or Survivor',185:'Jointly Or Severally',186:'Anyone Single Or Survivor',187:'Any Two Jointly'
#        ,189:'All Jointly',190:'Only First',192:'No 1,2,3 Jointly',197:'Authorised Signatory',199:'Managing Director',207:'Minor And Natural Guardian',208:'Minor And Legal Guardian',209:'Minor Alone',912:'NA',932:'NA'}
# df2['MODE_OF_OPERATION']=df2['MODE_OF_OPERATION'].map(mode)

# status = {'C':'Closed', 'A':'Active', 'H':'Hold', 'D':'Defreezed', 'T':'Amountwise freezed', 'F':'Freezed'}
# df2['STATUS_CODE']=df2['STATUS_CODE'].map(status)

# member = {'N':'NOT APPLI', 'S':'SHARE HOLDER', 'L':'NOMINAL MEMBER'}
# df2['MEMBER_FLAG']=df2['MEMBER_FLAG'].map(member)

# form = {'2':'PAN', '3':'FORM 60', '4':'FORM 61', '5':'NOT REQUIRED', '0':'Not Appli'}
# df2['FORM_60_FLAG']=df2['FORM_60_FLAG'].map(form)

# credit = {'N':'NO', 'Y':'YES'}
# df2['CREDIT_CARD_FLAG']=df2['CREDIT_CARD_FLAG'].map(credit)

# debit = {'N':'NO', 'Y':'YES','R':'R'}
# df2['DEBIT_CARD_FLAG']=df2['DEBIT_CARD_FLAG'].map(debit)

# sms = {'N':'NO', 'Y':'YES'}
# df2['SMS_BANKING_FLAG']=df2['SMS_BANKING_FLAG'].map(sms)

# mob = {'N':'NO', 'Y':'YES'}
# df2['MOBILE_BANKING_FLAG']=df2['MOBILE_BANKING_FLAG'].map(mob)

# net = {'N':'NO', 'Y':'YES'}
# df2['NET_BANKING_FLAG']=df2['NET_BANKING_FLAG'].map(net)

# nri = {'N':'NO', 'Y':'YES'}
# df2['NRE_NRO_NRI_FLAG']=df2['NRE_NRO_NRI_FLAG'].map(nri)

# kyc = {'N':'NO', 'Y':'YES'}
# df2['KYC_COMPLETE_FLAG']=df2['KYC_COMPLETE_FLAG'].map(kyc)

# lic = {'N':'NO', 'Y':'YES'}
# df2['ESTABLISH_LICENCE_FLAG']=df2['ESTABLISH_LICENCE_FLAG'].map(lic)

# def func(GST_VERIFY):
#     if GST_VERIFY=='N':
#         return 'NO'
#     elif GST_VERIFY=='Y':
#         return 'YES'
#     return 'None'
# df2['GST_VERIFY_FLAG']=df2['GST_VERIFY_FLAG'].apply(func)

# house = {'N':'NO', 'Y':'YES'}
# df2['HOUSE_OWN']=df2['HOUSE_OWN'].map(house)

# car = {'N':'NO', 'Y':'YES'}
# df2['CAR_MOTOR_OWN']=df2['CAR_MOTOR_OWN'].map(car)

# gl = {'172':'GOLD LOAN','177':'STAFF HOUSING LOAN','167':'CASH CREDIT GOLD LOAN A/C','186':'DOCUMENTARY BILL PURCHASE',
# '174':'HYPHOTICATION LOAN','178':'ADVANCE AGAINST FIXED DEPOSIT','180':'ADVANCE AGAINST RECURRING DEPOSIT',
# '176':'STAFF LOAN','171':'HIRE PURCHASE LOAN','170':'HOUSE LOAN ACCOUNT','168':'CASH CREDIT TERM LOAN','169':'MORTAGUAGE LOAN',
# '179':'ADVANCE AGAINST PIGMY DEPOSIT','173':'INDIVIDUAL CLEAN CASH CREDIT','182':'ADVANCE AGAINST WAREHOUSE RECEIPTS',
# '185':'INLAND BILL DISCOUNTED','184':'ADVANCE AGAINST GOVT. SECURITY','175':'INDIVIDUAL PLEDGE OF GOOD',
# '181':'ADVANCE AGAINST REINVESTMENT RECEIPTS'}
# df2['GLCODE']=df2['GLCODE'].map(gl)

# actype = {449:'Internal Account', 450:'Control Account'}
# df2['ACCOUNT_TYPE_ID']=df2['ACCOUNT_TYPE_ID'].map(actype)

# acsecure = {'N':'NO', 'Y':'YES'}
# df2['AC_SECURED_FLAG']=df2['AC_SECURED_FLAG'].map(acsecure)

# add = {'N':'NO', 'A':'Additional'}
# df2['ADDITIONAL_YN']=df2['ADDITIONAL_YN'].map(add)

# def func(CASH_SECURITY):
#     if CASH_SECURITY=='N':
#         return 'NO'
#     elif CASH_SECURITY=='Y':
#         return 'YES'
#     return 'None'
# df2['CASH_SECURITY_FLAG']=df2['CASH_SECURITY_FLAG'].apply(func)

# lien = {'N':'NO', 'Y':'YES'}
# df2['LIEN_FLAG']=df2['LIEN_FLAG'].map(lien)

# npa = {'N':'NO', 'Y':'YES'}
# df2['NPA_FLAG']=df2['NPA_FLAG'].map(npa)

# renew = {'N':'NO', 'Y':'YES'}
# df2['RENEWAL_YN']=df2['RENEWAL_YN'].map(renew)

# subsidy = {'N':'NO', 'Y':'YES'}
# df2['SUBSIDY_APPLIED_FLAG']=df2['SUBSIDY_APPLIED_FLAG'].map(subsidy)





