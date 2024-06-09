from django import forms
from .models import CSVUpload

class CSVUploadForm(forms.ModelForm):
    class Meta:
        model = CSVUpload
        fields = ['file']
