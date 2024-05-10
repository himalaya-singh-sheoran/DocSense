import unittest
from unittest.mock import patch, MagicMock
from app import SharePointDataloader

class TestSharePointDataloader(unittest.TestCase):
    @patch('app.SharePointDataloader.openpyxl.load_workbook')
    def test_yield_data_from_xlsx_success(self, mock_load_workbook):
        # Create a mock worksheet with sample data
        mock_worksheet = MagicMock()
        mock_worksheet.iter_rows.return_value = [
            ('value11', 'value12'),
            ('value21', 'value22'),
            ('value31', 'value32'),
            ('value41', 'value42'),
            ('value51', 'value52'),
            ('value61', 'value62'),
        ]

        # Create a mock workbook and set its active sheet
        mock_workbook = MagicMock()
        mock_workbook.active = mock_worksheet

        # Configure the mock load_workbook to return the mock workbook
        mock_load_workbook.return_value = mock_workbook

        # Create an instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")

        # Call yield_data_from_xlsx method
        file_relative_url = "test.xlsx"
        batch_size = 3
        data_generator = sharepoint_dataloader.yield_data_from_xlsx(file_relative_url, batch_size)

        # Assert that the generator yields the expected data
        expected_data = [
            [{'column1': 'value11', 'column2': 'value12'}, {'column1': 'value21', 'column2': 'value22'}, {'column1': 'value31', 'column2': 'value32'}],
            [{'column1': 'value41', 'column2': 'value42'}, {'column1': 'value51', 'column2': 'value52'}, {'column1': 'value61', 'column2': 'value62'}],
        ]
        self.assertEqual(list(data_generator), expected_data)

    @patch('app.SharePointDataloader.openpyxl.load_workbook')
    def test_yield_data_from_xlsx_failure(self, mock_load_workbook):
        # Mock load_workbook function to raise an exception
        mock_load_workbook.side_effect = Exception("Failed to load workbook")

        # Create an instance of SharePointDataloader
        sharepoint_dataloader = SharePointDataloader(connector_name="sharepoint", connector_configs="TEST_CONFIG")

        # Call yield_data_from_xlsx method and assert that the expected exception is raised
        file_relative_url = "test.xlsx"
        batch_size = 3
        with self.assertRaises(Exception) as context:
            sharepoint_dataloader.yield_data_from_xlsx(file_relative_url, batch_size)
        self.assertEqual(str(context.exception), "Error when yielding data from xlxs file: Exception Failed to load workbook")

if __name__ == '__main__':
    unittest.main()
