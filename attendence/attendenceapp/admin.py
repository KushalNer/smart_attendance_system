from django.contrib import admin
from .models import Teacher, Subject, Student, Allattendance
# Register your models here.
admin.site.register(Teacher)
admin.site.register(Subject)
admin.site.register(Student)
admin.site.register(Allattendance)
