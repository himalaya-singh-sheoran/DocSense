import unittest
from unittest.mock import MagicMock, patch
from app import SharePointDataloader

class TestSharePointDataloader(unittest.TestCase):
    @patch('app.connectors.sharepoint.connector.SharePointConnector')
    @patch('os.getenv', return_value="{'username': 'test_user', 'password': 'test_pass', 'site_url': 'http://example.com'}")
    def test_init(self, mock_os_getenv, mock_sharepoint_connector):
        # Test initialization with SharePoint connector
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        mock_os_getenv.assert_called_once_with("TEST_CONFIG")
        mock_sharepoint_connector.assert_called_once_with(username='test_user', password='test_pass', site_url='http://example.com')
    
    @patch.object(SharePointDataloader, 'spc')
    def test_download_xlxs_files_success(self, mock_spc):
        # Test successful download of xlsx file
        mock_web = MagicMock()
        mock_file = MagicMock()
        mock_download_session = MagicMock()
        mock_download_session.execute_query.return_value = None
        mock_file.download_session.return_value = mock_download_session
        mock_web.get_file_by_server_relative_url.return_value = mock_file
        mock_ctx = MagicMock()
        mock_ctx.web.return_value = mock_web
        mock_spc.establish_sharepoint_context.return_value = mock_ctx
        
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        sharepoint_dataloader.download_xlxs_files("file_relative_url")
        mock_file.download_session.assert_called_once()
    
    @patch.object(SharePointDataloader, 'spc')
    def test_download_xlxs_files_failure(self, mock_spc):
        # Test failure to download xlsx file
        mock_web = MagicMock()
        mock_web.get_file_by_server_relative_url.side_effect = Exception("Failed to get file")
        mock_ctx = MagicMock()
        mock_ctx.web.return_value = mock_web
        mock_spc.establish_sharepoint_context.return_value = mock_ctx
        
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        with self.assertRaises(Exception) as context:
            sharepoint_dataloader.download_xlxs_files("file_relative_url")
        self.assertEqual(str(context.exception), "Unable to download xlsx from sharepoint: Exception Failed to get file")
    
if __name__ == '__main__':
    unittest.main()
        
