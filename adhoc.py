import unittest
from unittest.mock import MagicMock, patch
from your_module import SharePointDataloaderApi

class TestSharePointDataloaderApi(unittest.TestCase):
    def setUp(self):
        # Mocking environment variable containing a string representation of dictionary
        with patch.dict('os.environ', {'test_config': "{'client_id': '123', 'client_secret': '456'}"}):
            self.sharepoint_api = SharePointDataloaderApi(
                connector_name="sharepoint_api",
                connector_configs="test_config",
                site_url="http://example.com"
            )

    @patch('requests.Session.get')
    def test_yield_data_from_sharepoint_list_success(self, mock_get):
        # Mocking successful responses for SharePoint API requests
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {'d': {'results': [1, 2, 3]}, '__next': 'http://example.com/next_page'}
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {'d': {'results': [4, 5, 6]}}  # Last page
        mock_get.side_effect = [mock_response1, mock_response2]

        # Test the function with a valid list title and batch size
        data_generator = self.sharepoint_api.yield_data_from_sharepoint_list("TestList", 3)
        data = [item for sublist in data_generator for item in sublist]
        self.assertEqual(data, [1, 2, 3, 4, 5, 6])

    @patch('requests.Session.get')
    def test_yield_data_from_sharepoint_list_empty_response(self, mock_get):
        # Mocking an empty response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'d': {'results': []}}  # Empty page
        mock_get.return_value = mock_response

        # Test the function with a valid list title and batch size
        data_generator = self.sharepoint_api.yield_data_from_sharepoint_list("TestList", 3)
        data = [item for sublist in data_generator for item in sublist]
        self.assertEqual(data, [])

    @patch('requests.Session.get')
    def test_yield_data_from_sharepoint_list_failure(self, mock_get):
        # Mocking a failure response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {'error': 'List not found'}
        mock_get.return_value = mock_response

        # Test the function with a non-existent list title
        with self.assertRaises(Exception) as context:
            data_generator = self.sharepoint_api.yield_data_from_sharepoint_list("NonExistentList", 3)
            next(data_generator)
        self.assertEqual(str(context.exception), "Failed to get data from sharepoint list. Api Response: {'error': 'List not found'}")

    @patch('requests.Session.get')
    def test_yield_data_from_sharepoint_list_exception(self, mock_get):
        # Mocking an exception raised during the request
        mock_get.side_effect = Exception("Connection error")

        # Test the function when an exception occurs during the request
        with self.assertRaises(Exception) as context:
            data_generator = self.sharepoint_api.yield_data_from_sharepoint_list("TestList", 3)
            next(data_generator)
        self.assertEqual(str(context.exception), "Error in class SharePointDataloaderApi in function yield_data_from_sharepoint_list: <class 'Exception'>")

if __name__ == '__main__':
    unittest.main()
    
