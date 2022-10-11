import os
import sys
import argparse
import subprocess
import random
import string

parser = argparse.ArgumentParser()
parser.add_argument('--bucket', dest='bucket_required', action='store_true', 
                    default=False, help='Bucket required')
parser.add_argument('--email', dest='email_required', action='store_true', 
                    default=False, help='Email required')
parser.add_argument('--packages', dest='extra_packages',
                    default='', type=str, help='additional required packages')
args = parser.parse_args()

extra_pkgs = args.extra_packages


# Installation


# The Vertex AI Workbench Notebook product has specific requirements
IS_WORKBENCH_NOTEBOOK = os.getenv("DL_ANACONDA_HOME")
IS_USER_MANAGED_WORKBENCH_NOTEBOOK = os.path.exists(
    "/opt/deeplearning/metadata/env_version"
)
IS_COLAB = "google.colab" in sys.modules

# Vertex AI Notebook requires dependencies to be installed with '--user'
USER_FLAG = ""
if IS_WORKBENCH_NOTEBOOK:
    USER_FLAG = "--user"
    
# not used
'''
print("Installing packages")
os.system(f"pip3 install --upgrade --quiet {USER_FLAG} google-cloud-aiplatform {args.extra_packages}")
print("Done installation")
'''

# Authenticate
if IS_COLAB:
    from google.colab import auth as google_auth

    google_auth.authenticate_user()


# project ID
if IS_WORKBENCH_NOTEBOOK:
    shell_output = subprocess.check_output("gcloud config list --format 'value(core.project)' 2>/dev/null", shell=True)
    PROJECT_ID = shell_output[0:-1].decode('utf-8')
    print("PROJECT ID: ", PROJECT_ID)
else:
    PROJECT_ID = input("Enter PROJECT_ID: ")
    os.system(f"gcloud config set project {PROJECT_ID}")

# email
if args.email_required:
    if IS_WORKBENCH_NOTEBOOK:
        shell_output = subprocess.check_output("gcloud config list --format 'value(core.account)' 2>/dev/null", shell=True)
        EMAIL_ADDR = shell_output[0:-1].decode('utf-8')
        print("EMAIL_ADDR: ", EMAIL_ADDR)
    else:
        EMAIL_ADDR = input("Enter Email Address: ")

# region
shell_output = subprocess.check_output("gcloud config list --format 'value(ai.region)'", shell=True)
REGION = shell_output[0:-1].decode('utf-8')
if REGION == '':
    REGION = input("Enter REGION: ")
print("REGION: ", REGION)

# multi-region
MULTI_REGION = REGION.split('-')[0]

    
# UUID
# Generate a uuid of a specifed length(default=8)
def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


UUID = generate_uuid()
print("UUID", UUID)
    
# Bucket
if args.bucket_required:
    BUCKET_NAME = PROJECT_ID + "aip-" + UUID
    BUCKET_URI = f"gs://{BUCKET_NAME}"
    os.system(f"gsutil mb -l {REGION} {BUCKET_URI}")
    print("BUCKET_URI", BUCKET_URI)
    
    
# Project Number

if IS_WORKBENCH_NOTEBOOK:
    shell_output = subprocess.check_output("gcloud auth list 2>/dev/null", shell=True)
    SERVICE_ACCOUNT = shell_output[:-1].decode('utf-8').split('\n')[2].strip()
    PROJECT_NUMBER = SERVICE_ACCOUNT.split('-')[0]
else:
    shell_output = subprocess.check_output(f"gcloud projects describe {PROJECT_ID}", shell=True)
    PROJECT_NUMBER = shell_output[:-1].decode('utf-8').split('\n')[8].strip().replace("'", "")
    SERVICE_ACCOUNT = f"{PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
    
print("SERVICE_ACCOUNT", SERVICE_ACCOUNT)
print("PROJECT_NUMBER", PROJECT_NUMBER)

