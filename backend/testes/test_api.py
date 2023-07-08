import os
import sys
from dotenv import load_dotenv
# Load the environment variables from the .env file
load_dotenv()
# Get the base directory from the environment variable
base_dir = os.environ['BASE_DIR']
sys.path.append(f'{base_dir}/backend')
from app import app
import unittest

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_vul_lines_endpoint(self):
        response = self.app.get('/api/vul_lines?github_link=https://github.com/naemazam/Hotel-Management-System')
        self.assertEqual(response.status_code, 200)
        
        expected_result = {
            "ai_analysis": {
                "/HMSC.c": {
                    "file_url": "https://raw.githubusercontent.com/naemazam/Hotel-Management-System/main/HMSC.c",
                    "type_file": "c",
                    "vul_lines": [[21, 16, 19, 23, 17, 14, 15, 25, 26, 20]]
                }
            },
            "static_analysis": {}
        }
        
        response_data = response.get_json()
        self.assertEqual(response_data, expected_result)
        self.assertEqual(response.status_code, 200)

    def test_vul_lines_endpoint_invalid(self):
        response = self.app.get('/api/vul_lines?github_link=https://github.com/naemazam/Hotel-Mana')
        self.assertEqual(response.status_code, 400)
        
        expected_result = {
            'error': 'Invalid GitHub link'
        }
        
        response_data = response.get_json()
        self.assertEqual(response_data, expected_result)
        self.assertEqual(response.status_code, 400)    

    def test_vul_lines_endpoint_with_no_vul(self):
        response = self.app.get('/api/vul_lines?github_link=https://github.com/nixrajput/char-counter-cpp')
        self.assertEqual(response.status_code, 200)
        
        expected_result = {
            "ai_analysis": {},
            "static_analysis": {}
        }
        
        response_data = response.get_json()
        self.assertEqual(response_data, expected_result)
        self.assertEqual(response.status_code, 200)    

if __name__ == '__main__':
    unittest.main()