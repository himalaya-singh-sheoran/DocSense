import requests
from requests.auth import HTTPBasicAuth

# SharePoint site URL
site_url = 'https://yourtenant.sharepoint.com/sites/yoursite'

# Client ID and Client Secret obtained from SharePoint App registration
client_id = 'your_client_id'
client_secret = 'your_client_secret'

# SharePoint list name
list_name = 'your_list_name'

# SharePoint REST API URL to get items from the list
api_url = f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items"

# Authentication endpoint
auth_url = f"{site_url}/_layouts/15/oauth2/token"

# Payload for authentication
auth_data = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
    'resource': site_url
}

# Getting access token
response = requests.post(auth_url, data=auth_data)
access_token = response.json()['access_token']

# Request headers
headers = {
    'Authorization': f'Bearer {access_token}',
    'Accept': 'application/json;odata=verbose'
}

# Make request to get data from SharePoint list
response = requests.get(api_url, headers=headers)

# Check if request was successful
if response.status_code == 200:
    data = response.json()
    # Process the data as needed
    print(data)
else:
    print(f"Failed to retrieve data. Error: {response.text}")
