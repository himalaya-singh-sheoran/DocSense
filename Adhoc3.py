import msal
import requests

# Azure AD app registration details
tenant_id = 'your-tenant-id'
client_id = 'your-client-id'
client_secret = 'your-client-secret'

# SharePoint site and list details
site_domain = 'yourtenant.sharepoint.com'
site_path = '/sites/yoursite'
list_name = 'Your List Name'

# Acquire token
authority_url = f'https://login.microsoftonline.com/{tenant_id}'
app = msal.ConfidentialClientApplication(
    client_id,
    authority=authority_url,
    client_credential=client_secret
)

scope = ['https://graph.microsoft.com/.default']
result = app.acquire_token_for_client(scopes=scope)

if 'access_token' in result:
    access_token = result['access_token']
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }

    # Construct the site URL
    site_url = f'https://graph.microsoft.com/v1.0/sites/{site_domain}:{site_path}'

    # Get the site ID
    site_response = requests.get(site_url, headers=headers)
    if site_response.status_code == 200:
        site_data = site_response.json()
        site_id = site_data['id']

        # Get the list ID by list name
        lists_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/lists'
        lists_response = requests.get(lists_url, headers=headers)
        if lists_response.status_code == 200:
            lists_data = lists_response.json()
            list_id = None
            for lst in lists_data.get('value', []):
                if lst['name'] == list_name:
                    list_id = lst['id']
                    break

            if list_id:
                # Retrieve list items
                items_url = f'https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items?expand=fields'
                items_response = requests.get(items_url, headers=headers)
                if items_response.status_code == 200:
                    items = items_response.json().get('value', [])
                    for item in items:
                        print(item['fields'])
                else:
                    print(f"Error retrieving list items: {items_response.status_code}")
            else:
                print(f"List '{list_name}' not found.")
        else:
            print(f"Error retrieving lists: {lists_response.status_code}")
    else:
        print(f"Error retrieving site ID: {site_response.status_code}")
else:
    print(f"Error acquiring token: {result.get('error')}")
    print(result.get('error_description'))


