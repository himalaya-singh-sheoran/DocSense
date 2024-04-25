from msal import ConfidentialClientApplication
import requests

# Azure AD app configurations
client_id = 'your_client_id'
client_secret = 'your_client_secret'
tenant_id = 'your_tenant_id'
scope = ["https://graph.microsoft.com/.default"]

# SharePoint API endpoint
site_url = 'https://yourdomain.sharepoint.com/sites/yoursite'
list_name = 'your_list_name'
endpoint = f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items"

# Initialize MSAL app
app = ConfidentialClientApplication(
    client_id, authority=f"https://login.microsoftonline.com/{tenant_id}",
    client_credential=client_secret
)

# Get access token
result = app.acquire_token_for_client(scopes=scope)
access_token = result['access_token']

# Make request to SharePoint API
headers = {
    'Authorization': 'Bearer ' + access_token,
    'Content-Type': 'application/json'
}
response = requests.get(endpoint, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)  # Handle SharePoint data here
else:
    print("Error:", response.status_code, response.text)
