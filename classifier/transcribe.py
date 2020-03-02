import boto3
import time
import urllib
import json
from .config_aws import *
from botocore.exceptions import NoCredentialsError

def upload_to_aws(local_file, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, 
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    try:
        s3.upload_file(local_file, BUCKET, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def remove_from_aws(s3_file):
    
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, 
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    try:
        paginator = s3.get_paginator('list_object_versions')
        response_iterator = paginator.paginate(Bucket=BUCKET)
        for response in response_iterator:
            versions = response.get('Versions', [])
            versions.extend(response.get('DeleteMarkers', []))
            for version_id in [x['VersionId'] for x in versions
                            if x['Key'] == s3_file and x['VersionId'] != 'null']:
                print('Deleting {} version {}'.format(s3_file, version_id))
                s3.delete_object(Bucket=BUCKET, Key=s3_file, VersionId=version_id)
        print("Delete Successful")
        return True

    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

def transcribe_aws(audio_name):
    job_uri = S3_BUCKET_URL + audio_name
    job_name = 'textta-' + audio_name[:-4]

    transcribe = boto3.client('transcribe', 
                            aws_access_key_id=AWS_ACCESS_KEY_ID, 
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                            region_name=REGION)

    transcribe.start_transcription_job(TranscriptionJobName=job_name, 
                                        Media={'MediaFileUri': job_uri}, 
                                        MediaFormat='wav', 
                                        LanguageCode='id-ID')

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']: break
        print("Not ready yet...")
        time.sleep(2)
    # print(status)

    # transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        response = urllib.request.urlopen(status['TranscriptionJob']['Transcript']['TranscriptFileUri'])
        data = json.loads(response.read())
        text = data['results']['transcripts'][0]['transcript']
        print(text)
        return text
    else: return 'Gagal'

