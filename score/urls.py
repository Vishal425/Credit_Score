
from django.urls import path
from . import views
#from .views import GeneratePDF


urlpatterns = [
    path('cust_id/',views.cust_id),
    path('cust_id/test/', views.test,name='test'),#Calling  cust_id function to test function 
    path('database/',views.process),
    path('database/add/',views.add,name='add'),



    # path('emp/', views.emp),  
    # path('show/',views.show),  
    # path('emp/edit/<int:id>', views.edit),  
    # path('emp/update/<int:id>', views.update),  
    # path('emp/delete/<int:id>', views.destroy), 



    #path('credit_score/',views.credit_result,name='credit_result')
    #path('cust_id/pdf/',GeneratePDF.as_view(),name='GeneratePDF'),
]


