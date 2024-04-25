from shareplum import Site
from shareplum import Office365
from requests_ntlm import HttpNtlmAuth

# SharePoint site URL
site_url = "https://yourcompany.sharepoint.com/sites/yoursite"

# SharePoint client ID and client secret
client_id = "your_client_id"
client_secret = "your_client_secret"

# Connect to SharePoint site using client credentials
authcookie = Office365(site_url, client_id=client_id, client_secret=client_secret).GetCookies()

# Connect to SharePoint site using the authentication cookie
site = Site(site_url, auth=authcookie)

# Get a SharePoint list by title
list_name = "YourListName"
sp_list = site.List(list_name)

# Get all items from the list
items = sp_list.GetListItems()

# Print item titles
for item in items:
    print(item["Title"])
