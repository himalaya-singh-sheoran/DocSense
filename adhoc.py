import requests
from requests.auth import HTTPBasicAuth

# SharePoint site URL and list name
site_url = "https://your-sharepoint-site-url"
list_name = "YourListName"

# Client credentials
client_id = "YourClientId"
client_secret = "YourClientSecret"

# SharePoint API endpoint for retrieving list items
api_endpoint = f"{site_url}/_api/web/lists/getbytitle('{list_name}')/items"

# Authentication token request URL
auth_url = f"{site_url}/_layouts/15/OAuthAuthorize.aspx?client_id={client_id}&response_type=code&scope=Web.Write&redirect_uri=https://localhost"

# Fetching access token using client credentials
auth_response = requests.post(auth_url, auth=HTTPBasicAuth(client_id, client_secret))
if auth_response.status_code == 200:
    access_token = auth_response.json().get("access_token")
    # Making API call to SharePoint with access token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json;odata=verbose",
    }
    response = requests.get(api_endpoint, headers=headers)
    if response.status_code == 200:
        data = response.json()
        # Process retrieved items
        items = data['d']['results']
        for item in items:
            print(item)
    else:
        print("Error:", response.status_code)
else:
    print("Authentication failed:", auth_response.status_code)
