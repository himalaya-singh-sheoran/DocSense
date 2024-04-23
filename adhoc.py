from shareplum import Site
from shareplum import Office365
import pandas as pd

# SharePoint credentials
sharepoint_url = "https://your-sharepoint-site.sharepoint.com/sites/your-site"
username = "your_username@your_domain.com"
password = "your_password"

# Connect to SharePoint site
authcookie = Office365(sharepoint_url, username=username, password=password).GetCookies()
site = Site(sharepoint_url, authcookie=authcookie)

# Specify the SharePoint list name
list_name = "YourListName"

# Get data from the SharePoint list
data = site.List(list_name).GetListItems()

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Stream the DataFrame
print(df)
