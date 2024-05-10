import unittest
from unittest.mock import MagicMock, patch
from app import SharePointDataloader

class TestSharePointDataloader(unittest.TestCase):
    @patch('os.getenv', return_value="{'username': 'test_user', 'password': 'test_pass', 'site_url': 'http://example.com'}")
    def test_init(self, mock_os_getenv):
        # Test initialization with SharePoint connector
        with patch('app.connectors.sharepoint.connector.SharePointConnector') as mock_sharepoint_connector:
            # Mock the SharePointConnector class
            mock_sharepoint_connector_instance = MagicMock()
            mock_sharepoint_connector.return_value = mock_sharepoint_connector_instance
            
            # Create an instance of SharePointDataloader
            sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
            
            # Check if SharePointConnector is called with the correct arguments
            mock_sharepoint_connector.assert_called_once_with(username='test_user', password='test_pass', site_url='http://example.com')
            
            # Check if the self.spc attribute is set correctly
            self.assertEqual(sharepoint_dataloader.spc, mock_sharepoint_connector_instance)
    
    def test_download_xlxs_files_success(self):
        # Test successful download of xlsx file
        with patch.object(SharePointDataloader, 'spc') as mock_spc:
            # Mock the establish_sharepoint_context method
            mock_establish_sharepoint_context = MagicMock()
            mock_spc_instance = MagicMock()
            mock_spc_instance.establish_sharepoint_context = mock_establish_sharepoint_context
            mock_spc.return_value = mock_spc_instance
            
            # Mock the download_session method
            mock_download_session = MagicMock()
            mock_download_session.execute_query.return_value = None
            mock_file = MagicMock()
            mock_file.download_session.return_value = mock_download_session
            mock_web = MagicMock()
            mock_web.get_file_by_server_relative_url.return_value = mock_file
            mock_establish_sharepoint_context.return_value.web.return_value = mock_web
            
            # Create an instance of SharePointDataloader
            sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
            
            # Call the download_xlxs_files method
            sharepoint_dataloader.download_xlxs_files("file_relative_url")
            
            # Check if download_session is called
            mock_file.download_session.assert_called_once()
    
    def test_download_xlxs_files_failure(self):
        # Test failure to download xlsx file
        with patch.object(SharePointDataloader, 'spc') as mock_spc:
            # Mock the establish_sharepoint_context method to raise an exception
            mock_establish_sharepoint_context = MagicMock(side_effect=Exception("Failed to get file"))
            mock_spc_instance = MagicMock()
            mock_spc_instance.establish_sharepoint_context = mock_establish_sharepoint_context
            mock_spc.return_value = mock_spc_instance
            
            # Create an instance of SharePointDataloader
            sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
            
            # Check if the expected exception is raised
            with self.assertRaises(Exception) as context:
                sharepoint_dataloader.download_xlxs_files("file_relative_url")
            self.assertEqual(str(context.exception), "Unable to download xlsx from sharepoint: Exception Failed to get file")

if __name__ == '__main__':
    unittest.main()
    
