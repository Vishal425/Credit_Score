# import cx_Oracle
# import configs
# import pandas as pd
# connection = None
# try:
#     connection = cx_Oracle.connect(
#         config.username,
#         config.password,
#         config.dsn,
#         encoding=config.encoding)

#     # show the version of the Oracle Database
#     print(connection.version)
#     cur = connection.cursor()
#     # query = 'select * from py_customer_all ca \
#     #     full join py_cust_general g on ca.py_customer_id=g.py_customer_id \
#     #     full join py_customer_mast m on ca.py_customer_id=m.py_customer_id \
#     #     where ca.py_customer_id=325'
#     query = 'select * from py_customer_all ca \
#         inner join py_cust_general g on ca.py_customer_id=g.py_customer_id \
#         inner join py_customer_mast m on ca.py_customer_id=m.py_customer_id\
#         where ca.py_customer_id in (select py_customer_id from py_customer_mast where rownum < 1001)'
#     df = pd.read_sql_query(query, connection)
#     print('data feach')
#     df.to_csv('combine_1000_cust.csv',index=False)
#     print('database save into csv')
#     #print(df.head(5))
#     # list_col=list(df.columns)
#     # print(list_col)

#     #Check the email address of the customer 
#     # import re
#     # regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
#     # email=df['EMAIL_ID'].unique()
#     # if email =='None'or 'NA':
#     #     df['EMAIL_ID_Y_N']=0
#     # elif (re.search(regex, email)):
#     #     df['EMAIL_ID_Y_N']=1
#     # else:
#     #     df['EMAIL_ID_Y_N']=0
    
   
#     # # #Check the mobile number is valid or not 
#     # Pattern = re.compile("(0/91)?[7-9][0-9]{9}")
#     # mobile=df['MOBILE_NO'].unique()
#     # print(mobile)
#     # if mobile == 'None' or 'NA':
#     #     df['MOBILE_NO_Y_N']=0
#     # elif Pattern.match(mobile):
#     #     df['MOBILE_NO_Y_N']=1
#     # else:
#     #     df['MOBILE_NO_Y_N']=0
    
#     # print(df['MOBILE_NO_Y_N'])

#     #print(df)


   


  

# except cx_Oracle.Error as error:
#     print('This is  the error:',error)
# finally:
#     # release the connection
#     if connection:
#         connection.close()


