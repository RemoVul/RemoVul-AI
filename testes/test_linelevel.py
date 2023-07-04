import unittest
import numpy as np
from dotenv import load_dotenv
import os
import sys
import torch
# Load the environment variables from the .env file
load_dotenv()
# Get the base directory from the environment variable
base_dir = os.environ['BASE_DIR']
sys.path.append(base_dir)
from removul.linelevel import remove_special_token_score, get_lines_score, get_evaluation_matrix, verify_flaw_line_in_func, summerize_attention


class TestFunctions(unittest.TestCase):
    def test_remove_special_token_score(self):
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        expected_result = [0, 0.2, 0.3, 0.4, 0]
        result = remove_special_token_score(scores)
        self.assertEqual(result, expected_result)

    def test_get_lines_score(self):
        word_score_list = [['int', 0.1], ['x', 0.2], ['=', 0.3], ['15', 0.4],['Ċ',0.2], ['int', 0.5], ['y', 0.6], ['=', 0.7], ['20', 0.8]]
        verified_flaw_lines = ['intx=15']
        expected_result = ([0.1+0.2+0.3+0.4+0.2, 0.5+0.6+0.7+0.8], [0])
        result = get_lines_score(word_score_list, verified_flaw_lines)
        self.assertEqual(result, expected_result)

    def test_get_evaluation_matrix(self):
        # score for each line is a list of tensors
        score_for_each_line = [torch.tensor(0.1+ 0.2+ 0.3+ 0.4+ 0.5), torch.tensor(0.1+ 0.2+ 0.3+ 0.4+ 0.5)]
        index_of_flaw_lines = [0]
        expected_result = (1, 1)
        result = get_evaluation_matrix(score_for_each_line, index_of_flaw_lines)
        self.assertEqual(result, expected_result)

    def test_verify_flaw_line_in_func(self):
        func_tokens = ["int", "x", "=", "15", "int", "y", "=", "20"]
        flow_tokens = [["int", "x", "=", "15"], ["int", "y", "=", "50"]]
        expected_result = (True, [["int", "x", "=", "15"]])
        result = verify_flaw_line_in_func(func_tokens, flow_tokens)
        self.assertEqual(result, expected_result)

    def test_summerize_attention(self):
        attentions = [torch.ones(1, 12, 512, 512)]
        #array of shape (512,) value is 6144
        expected_result = np.ones((512,)) * 6144
        result = summerize_attention(attentions)
        np.testing.assert_array_equal(result, expected_result)

if __name__ == '__main__':
    unittest.main()
