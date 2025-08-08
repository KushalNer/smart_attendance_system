from django.urls import path
from attendenceapp import views

urlpatterns = [
    path('', views.home, name="home"),
    path('registerteacher/', views.register_teacher, name="tregister"),
    path('registerstudent/', views.register_face, name="sregister"),
    path('userlogin/', views.user_login , name="loginaccount"),
    path('userlogout/',views.user_logout, name="userlogout"),
    path('subject/<str:username>/',views.subject, name="subjectdata"),
    path('delete_subject/<str:subject_name>/', views.delete_subject, name='delete_subject'),
    path('attendence/', views.user_attendence, name="attendence"),
    path('attendence_data/',views.attendence_data, name="attendencedata"),
    path('contactus/', views.contact_us, name="contactus"),
    path('analysis/', views.data_analysis ,name="dataanalysis"),
    path('attendence/<str:username>/', views.recognize, name="recognize_face"),
    path('userprofile/<str:username>/', views.user_profile, name="userprofile"),
    #extra functionality paths

    path('attendence_data/exportdatewise/', views.datewise_download , name='exportdatewise'),
    path('analysis/exportdata/',views.analysis_report_downlaod , name="analysis_report_download"),
    path('get-subject/',views.barchart, name="barchart")
    
]
