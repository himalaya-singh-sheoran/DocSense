import unittest
from unittest.mock import MagicMock, patch
from app import SharePointDataloader

class TestSharePointDataloader(unittest.TestCase):
    @patch('os.getenv', return_value="{'client_id': 'test_id', 'client_secret': 'test_secret', 'site_url': 'http://example.com'}")
    @patch('app.connectors.sharepoint.connector.SharePointConnector')
    def test_download_xlxs_files_success(self, mock_sharepoint_connector, mock_os_getenv):
        # Mock SharePointConnector class
        mock_sharepoint_connector_instance = MagicMock()
        mock_sharepoint_connector.return_value = mock_sharepoint_connector_instance
        
        # Mock SharePointConnector.establish_sharepoint_context method
        mock_client_context = MagicMock()
        mock_sharepoint_connector_instance.establish_sharepoint_context.return_value = mock_client_context
        
        # Mock ClientContext.web.get_file_by_server_relative_url().download_session().execute_query() method chain
        mock_file = MagicMock()
        mock_download_session = MagicMock()
        mock_execute_query = MagicMock()
        mock_execute_query.return_value = None
        mock_download_session.execute_query.return_value = mock_execute_query
        mock_file.download_session.return_value = mock_download_session
        mock_web = MagicMock()
        mock_web.get_file_by_server_relative_url.return_value = mock_file
        mock_client_context.web.return_value = mock_web
        
        # Create instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        
        # Call download_xlxs_files method
        sharepoint_dataloader.download_xlxs_files("file_relative_url")
        
        # Assert that methods were called
        mock_sharepoint_connector_instance.establish_sharepoint_context.assert_called_once()
        mock_web.get_file_by_server_relative_url.assert_called_once_with("file_relative_url")
        mock_file.download_session.assert_called_once()
        mock_download_session.execute_query.assert_called_once()
    
    @patch('os.getenv', return_value="{'client_id': 'test_id', 'client_secret': 'test_secret', 'site_url': 'http://example.com'}")
    @patch('app.connectors.sharepoint.connector.SharePointConnector')
    def test_download_xlxs_files_failure(self, mock_sharepoint_connector, mock_os_getenv):
        # Mock SharePointConnector class to raise exception
        mock_sharepoint_connector_instance = MagicMock()
        mock_sharepoint_connector.return_value = mock_sharepoint_connector_instance
        mock_sharepoint_connector_instance.establish_sharepoint_context.side_effect = Exception("Failed to establish context")
        
        # Create instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        
        # Assert that the expected exception is raised
        with self.assertRaises(Exception) as context:
            sharepoint_dataloader.download_xlxs_files("file_relative_url")
        self.assertEqual(str(context.exception), "Unable to download xlsx from sharepoint: Exception Failed to establish context")

if __name__ == '__main__':
    unittest.main()
    
