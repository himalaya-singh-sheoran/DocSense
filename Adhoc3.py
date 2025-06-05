import requests
from msal import ConfidentialClientApplication

# Replace these with your actual values
TENANT_ID = 'your-tenant-id'
CLIENT_ID = 'your-client-id'
CLIENT_SECRET = 'your-client-secret'
SHAREPOINT_SITE_NAME = 'your-site-name'           # e.g. 'contoso'
SHAREPOINT_SITE_HOST = 'yourdomain.sharepoint.com'
SHAREPOINT_DOC_LIB = 'Shared Documents'           # default library
FILE_PATH = 'Folder1/Folder2/filename.xlsx'       # path in doc library
DOWNLOAD_PATH = 'downloaded_file.xlsx'            # local path

# Get token using MSAL
authority_url = f'https://login.microsoftonline.com/{TENANT_ID}'
scope = ['https://graph.microsoft.com/.default']

app = ConfidentialClientApplication(
    CLIENT_ID,
    authority=authority_url,
    client_credential=CLIENT_SECRET
)

token_response = app.acquire_token_for_client(scopes=scope)

if "access_token" not in token_response:
    raise Exception("Failed to acquire token: " + str(token_response))

access_token = token_response['access_token']

# Get Site ID
site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE_HOST}:/sites/{SHAREPOINT_SITE_NAME}"
headers = {'Authorization': f'Bearer {access_token}'}
site_resp = requests.get(site_url, headers=headers)
site_resp.raise_for_status()
site_id = site_resp.json()['id']

# Get Drive ID (usually the default document library)
drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
drive_resp = requests.get(drive_url, headers=headers)
drive_resp.raise_for_status()
drive_id = drive_resp.json()['value'][0]['id']

# Get File ID
file_metadata_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{FILE_PATH}"
file_resp = requests.get(file_metadata_url, headers=headers)
file_resp.raise_for_status()
download_url = file_resp.json()['@microsoft.graph.downloadUrl']

# Download file
download_resp = requests.get(download_url)
with open(DOWNLOAD_PATH, 'wb') as f:
    f.write(download_resp.content)

print(f"✅ File downloaded successfully to: {DOWNLOAD_PATH}")
