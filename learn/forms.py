from django import forms
from django.core.exceptions import ValidationError

NGRAM_CHOICES = (((1, 1), 'Unigram'),
                 ((1, 2), 'Bigram'), 
                 ((1, 3), 'Trigram'), )

RESAMPLE_CHOICES = (('None', 'None'),
                 ('minority', 'Upsampling'), 
                 ('auto', 'Auto'), )

def validate_file_size(value):
    filesize= value.size
    if filesize > 2621440:
        raise ValidationError("The maximum file size that can be uploaded is 2.5 MB")
    else: return value

def validate_file_extension(value):
    filetype = value.content_type
    if filetype != "text/csv":
        raise ValidationError("Please upload only .csv extention file")
    else: return value

class pipelineForm(forms.Form):
    category = forms.CharField(label='Category', max_length=16)
    question = forms.CharField(label='Question', max_length=1024)
    dataset = forms.FileField(label='Dataset', validators=[validate_file_size, validate_file_extension])
    stopwords = forms.BooleanField(label='Stopwords', initial=False, required=False)
    stemming = forms.BooleanField(label='Stemming', initial=False, required=False)
    n_gram = forms.ChoiceField(label='N-Gram', choices=NGRAM_CHOICES, initial=NGRAM_CHOICES[1])
    resample = forms.ChoiceField(label="Resampling", choices=RESAMPLE_CHOICES, initial=RESAMPLE_CHOICES[0])

    # for bootstrap styling
    def __init__(self, *args, **kwargs):
        super(pipelineForm, self).__init__(*args, **kwargs)
        for visible in self.visible_fields():
            if hasattr(visible.field.widget, 'input_type'):
                if visible.field.widget.input_type in ['radio', 'checkbox']:
                    visible.field.widget.attrs['class'] = 'form-check-input'
                else:
                    visible.field.widget.attrs['class'] = 'form-control'
