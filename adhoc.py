import requests

# Azure AD app details
client_id = 'your_client_id'
client_secret = 'your_client_secret'
tenant_id = 'your_tenant_id'
resource = 'https://<your_domain>.sharepoint.com'

# SharePoint site details
site_url = 'https://<your_domain>.sharepoint.com/sites/<site_name>'
list_name = 'YourListName'

# Get OAuth2 token
token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/token"
payload = {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
    'resource': resource
}
response = requests.post(token_url, data=payload)
access_token = response.json().get('access_token')

if access_token:
    # SharePoint API endpoint for lists
    list_endpoint = f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items"

    # Make a GET request to retrieve items from the SharePoint list
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json;odata=verbose'
    }
    response = requests.get(list_endpoint, headers=headers)

    # Process the response
    if response.status_code == 200:
        data = response.json()
        items = data.get('d', {}).get('results', [])
        for item in items:
            print(item.get('Title'))  # Adjust the field name as per your SharePoint list schema
    else:
        print("Failed to retrieve items. Status code:", response.status_code)
else:
    print("Failed to obtain access token.")
