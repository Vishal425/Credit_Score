#https://www.javatpoint.com/django-crud-applicationS

import json
from django.shortcuts import render,redirect
from django.http import HttpResponse, JsonResponse
from jinja2 import Environment, FileSystemLoader
#from weasyprint import HTML

# Create your views here.
#from django.db import connection
from pandas.core.indexes.base import Index

#from score.test_sql_3 import *

#Creata the function for enter the customer_id 
def cust_id(request):
     return render(request,'score/cust_id.html')


# def test(request):
#     val1=int(request.GET['num1'])
#     json_Value=(sql_value(val1))

#     print(json_Value)
#     print(type(json_Value))
#     value=JsonResponse(sql_value(),safe=False)
#     print(type(value))
#     return value
 
    #return render(request,'score/test.html',{'json_Value':json_Value})
    #return HttpResponse(json_Value)
    #return JsonResponse(sql_value(),safe=False)
    #return ("Hellow world")




from django.db import connection

def process(request):
    #json_Value=(sql_value())
    # print(json_Value)
    # print(type(json_Value))
    #value=JsonResponse(json_Value,safe=False)
    # print(type(value))
    return render(request,'score/test1.html')#,{'json_Value':json_Value})
    # value_dict=json.loads(str(value))
    # print(type(value_dict))
    #return HttpResponse("hellow world")
  

from score.credit_score_model_making import *
def add(request):
    val1=int(request.GET['num1'])
    val2=int(request.GET['num2'])
    new=displayText(val1,val2)
    # res= val1+val2
    return render(request,'score/result.html',{'result':new})




from django.test import TestCase
from django.http import HttpResponse
from django.template.loader import get_template
from .pdf_converter import render_to_pdf
from django.views.generic import View
from score.credit_score_prediction_file import *

# Create your tests here.
def test(request):
    #template = get_template('score/report.html')
    val1=int(request.GET['num1'])
    print(val1)
    df=(sql_value(val1)) #calling other fuction using sending id number
    #print(df)
    
    # df = pd.read_csv(r'D:\Python_Project\py\credit_analysis\credit_score\TDataset\predicted_data.csv')

    json_Value=df.to_dict(orient='records')
           
    pdf = render_to_pdf('score/report.html',{'json_Values':json_Value})
    
    if pdf:
        response = HttpResponse(pdf, content_type='application/pdf')
        filename = "Invoice_%s.pdf" %("12341231")
        content = "inline; filename='%s'" %(filename)
        download = request.GET.get("download")
        if download:
            content = "attachment; filename='%s'" %(filename)
        response['Content-Disposition'] = content
        return response
    return HttpResponse("Not found")


################################ CRUD ###############################
from score.forms import EmployeeForm  
from score.models import Employee  
from django.shortcuts import render, redirect  
# Create your views here.  
def emp(request):  
    if request.method == "POST":  
        form = EmployeeForm(request.POST)  
        if form.is_valid():  
            try:  
                form.save()  
                return redirect('/show')  
            except:  
                pass  
    else:  
        form = EmployeeForm()  
    return render(request,'score/index.html',{'form':form})  
def show(request):  
    employees = Employee.objects.all()  
    return render(request,"score/show.html",{'employees':employees})  
def edit(request, id):  
    employee = Employee.objects.get(id=id)  
    return render(request,'score/edit.html', {'employee':employee})  
def update(request, id):  
    employee = Employee.objects.get(id=id)  
    form = EmployeeForm(request.POST, instance = employee)  
    if form.is_valid():  
        form.save()  
        return redirect("/show")  
    return render(request, 'score/edit.html', {'employee': employee})  
def destroy(request, id):  
    employee = Employee.objects.get(id=id)  
    employee.delete()  
    return redirect("/show")  

