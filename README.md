# textta
Text Classification of Job Applicant Interviews using SVM, KNN & MNB Methods

## Installation
1. Create virtual environment `virtualenv -p python3.6 env`.
2. Install requirements `pip install -r requirements.txt`.
3. Create `config_aws.py` in `classifier/` and add this code below.
```
AWS_ACCESS_KEY_ID = 'your aws access key id'
AWS_SECRET_ACCESS_KEY = 'your aws secret accsess key'
REGION = 'your ragion'
BUCKET = 'your s3 bucket name'
S3_BUCKET_URL = 'your s3 bucket url'
```
