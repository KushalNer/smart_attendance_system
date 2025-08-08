from django.db import models
from django.utils import timezone
from django.utils.text import slugify
from datetime import datetime
# Create your models here.
class Teacher(models.Model):
    name = models.CharField( max_length=50)
    email = models.EmailField()
    image = models.ImageField(upload_to='teacherimage/')
    contact = models.CharField(max_length=10)
    username= models.CharField(max_length=50 ,blank=True)
    password = models.CharField(max_length=50)
    date = models.DateField( default=timezone.now)

    def __str__(self):
        return self.name
    
class Subject(models.Model):
    name = models.CharField(max_length=50)
    username = models.CharField(max_length=50, blank=True)
    slug = models.SlugField(unique=True, blank=True)
    classname = models.CharField(max_length=30, null=True)
    def save(self, *args, **kwargs):
        if not self.slug:
            base_slug = slugify(self.name)
            slug = base_slug
            counter = 1
            while Subject.objects.filter(slug=slug).exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            self.slug = slug
        super().save(*args, **kwargs)


    def __str__(self):
        return self.name


class Student(models.Model):
    name = models.CharField(max_length=50)
    email = models.EmailField()
    Class = models.CharField(max_length=30)
    gender = models.CharField(max_length=10, blank=True)
    rno = models.CharField(max_length=20, null=True)
    contact = models.CharField(max_length=10, blank=True)
    # teacher_name = models.ForeignKey(Teacher, on_delete=models.SET_NULL , null=True)
    # subject_name = models.ForeignKey(Subject, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.email
    
class Allattendance(models.Model):
    aname = models.CharField(max_length=50, blank=True)
    aclass = models.CharField(max_length=30, blank=True)
    agender = models.CharField(max_length=10, blank=True)
    arno = models.CharField(max_length=20, null=True)
    adate = models.CharField(max_length=20 )
    atime = models.CharField(max_length=30, default = timezone.localtime)
    ateacher_name = models.CharField(max_length=50, blank=True, null=True)
    asubject_name = models.CharField(max_length=50, blank=True, null=True)

    def __str__(self):
        return f"{self.aname} ------ {self.asubject_name}  ------- {self.adate}"

