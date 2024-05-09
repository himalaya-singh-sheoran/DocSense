import unittest
from unittest.mock import MagicMock, patch
from your_module import SharePointDataloaderApi

class TestSharePointDataloaderApi(unittest.TestCase):
    def setUp(self):
        self.sharepoint_api = SharePointDataloaderApi(
            connector_name="sharepoint_api",
            connector_configs="test_config",
            site_url="http://example.com"
        )

    @patch('requests.request')
    def test_get_total_number_of_items_success(self, mock_request):
        # Mocking a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'d': {'ItemCount': 10}}
        mock_request.return_value = mock_response

        # Test the function with a valid list title
        count = self.sharepoint_api.get_total_number_of_items_in_sharepoint_list("TestList")
        self.assertEqual(count, 10)

    @patch('requests.request')
    def test_get_total_number_of_items_failure(self, mock_request):
        # Mocking a failure response (e.g., list not found)
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {'error': 'List not found'}
        mock_request.return_value = mock_response

        # Test the function with a non-existent list title
        with self.assertRaises(Exception) as context:
            self.sharepoint_api.get_total_number_of_items_in_sharepoint_list("NonExistentList")
        self.assertEqual(str(context.exception), "Failed to get item count. Api Response: {'error': 'List not found'}")

    @patch('requests.request')
    def test_get_total_number_of_items_exception(self, mock_request):
        # Mocking an exception raised during the request
        mock_request.side_effect = Exception("Connection error")

        # Test the function when an exception occurs during the request
        with self.assertRaises(Exception) as context:
            self.sharepoint_api.get_total_number_of_items_in_sharepoint_list("TestList")
        self.assertEqual(str(context.exception), "An error occurred: Connection error")

    @patch('requests.request')
    def test_get_total_number_of_items_missing_access_token(self, mock_request):
        # Mocking a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'d': {'ItemCount': 10}}
        mock_request.return_value = mock_response

        # Test the function without providing an access token (should use the connector to fetch it)
        with patch.object(self.sharepoint_api.connector, 'get_access_token', return_value="test_token"):
            count = self.sharepoint_api.get_total_number_of_items_in_sharepoint_list("TestList")
        self.assertEqual(count, 10)

if __name__ == '__main__':
    unittest.main()
