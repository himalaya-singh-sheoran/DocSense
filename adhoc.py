import unittest
from unittest.mock import patch, MagicMock
from app import SharePointDataloader

class TestSharePointDataloader(unittest.TestCase):
    @patch('openpyxl.load_workbook')
    def test_yield_data_from_xlsx_success(self, mock_load_workbook):
        # Create a mock Workbook object
        mock_workbook = MagicMock()
        mock_workbook.active = MagicMock()
        mock_workbook.active.iter_rows.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        mock_load_workbook.return_value = mock_workbook
        
        # Create an instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        
        # Call yield_data_from_xlsx method
        data_generator = sharepoint_dataloader.yield_data_from_xlsx("file_relative_url", batch_size=2)
        
        # Check the yielded data
        self.assertEqual(next(data_generator), [{'Column1': 1, 'Column2': 2, 'Column3': 3}, {'Column1': 4, 'Column2': 5, 'Column3': 6}])
        self.assertEqual(next(data_generator), [{'Column1': 7, 'Column2': 8, 'Column3': 9}])
        with self.assertRaises(StopIteration):
            next(data_generator)
        
        # Assert that load_workbook was called with the correct arguments
        mock_load_workbook.assert_called_once_with(filename="file_relative_url.split('/')[-1]", read_only=True, data_only=True, keep_links=False)
    
    @patch('openpyxl.load_workbook')
    def test_yield_data_from_xlsx_empty_file(self, mock_load_workbook):
        # Create a mock Workbook object
        mock_workbook = MagicMock()
        mock_workbook.active = MagicMock()
        mock_workbook.active.iter_rows.return_value = []
        mock_load_workbook.return_value = mock_workbook
        
        # Create an instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        
        # Call yield_data_from_xlsx method
        data_generator = sharepoint_dataloader.yield_data_from_xlsx("file_relative_url", batch_size=2)
        
        # Check the yielded data
        with self.assertRaises(StopIteration):
            next(data_generator)
        
        # Assert that load_workbook was called with the correct arguments
        mock_load_workbook.assert_called_once_with(filename="file_relative_url.split('/')[-1]", read_only=True, data_only=True, keep_links=False)
    
    @patch('openpyxl.load_workbook')
    def test_yield_data_from_xlsx_exception(self, mock_load_workbook):
        # Mock load_workbook to raise an exception
        mock_load_workbook.side_effect = Exception("Failed to load workbook")
        
        # Create an instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")
        
        # Assert that the expected exception is raised
        with self.assertRaises(Exception) as context:
            sharepoint_dataloader.yield_data_from_xlsx("file_relative_url", batch_size=2)
        self.assertEqual(str(context.exception), "Error when yielding data from xlxs file: Exception Failed to load workbook")
        
        # Assert that load_workbook was called with the correct arguments
        mock_load_workbook.assert_called_once_with(filename="file_relative_url.split('/')[-1]", read_only=True, data_only=True, keep_links=False)

if __name__ == '__main__':
    unittest.main()
    
