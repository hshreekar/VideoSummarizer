# from django import forms
# from .models import Document

# class DocumentForm(forms.ModelForm):
#     class Meta:
#         model = Document
#         # fields = ('videoFile', 'subtitleFile')
#         fields = ('videoFile', 'subtitleFile', 'bonusWordsFile', 'stigmaWordsFile' )
#     def __init__(self, *args, **kwargs):
#     	super(DocumentForm, self).__init__(*args, **kwargs)
#     	self.fields['bonusWordsFile'].required = False
#     	self.fields['stigmaWordsFile'].required = False
# from django import forms
# from .models import Document

# class DocumentForm(forms.ModelForm):
#     class Meta:
#         model = Document
#         fields = ('videoFile', 'subtitleFile' )

#     def __init__(self, *args, **kwargs):
#     	super(DocumentForm, self).__init__(*args, **kwargs)
from django import forms
from .models import Document

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('videoFile', 'subtitleFile', 'bonusWordsFile', 'stigmaWordsFile' )

    def __init__(self, *args, **kwargs):
    	super(DocumentForm, self).__init__(*args, **kwargs)
    	self.fields['bonusWordsFile'].required = False
    	self.fields['stigmaWordsFile'].required = False