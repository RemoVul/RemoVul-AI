import unittest
import os
import sys
from dotenv import load_dotenv
# Load the environment variables from the .env file
load_dotenv()
# Get the base directory from the environment variable
base_dir = os.environ['BASE_DIR']
sys.path.append(base_dir)
from removul.filevul import get_c_and_cpp_files, read_c_file, tokenize_c_function,is_start_of_function

class TestCFileFunctions(unittest.TestCase):
    def setUp(self):
        self.test_directory = './test_files'
        os.makedirs(self.test_directory, exist_ok=True)
        self.create_test_files()

    def tearDown(self):
        os.remove(os.path.join(self.test_directory, 'test1.c'))
        os.remove(os.path.join(self.test_directory, 'test2.cpp'))
        os.removedirs(self.test_directory)

    def create_test_files(self):
        # Create test C file
        c_code = '''
        #include <stdio.h>

        int main() {
            printf("Hello World!");
            return 0;
        }
        '''
        with open(os.path.join(self.test_directory, 'test1.c'), 'w') as f:
            f.write(c_code)

        # Create test C++ file
        cpp_code = '''
        #include <iostream>

        int main() {
            std::cout << "Hello World!" << std::endl;
            return 0;
        }
        '''
        with open(os.path.join(self.test_directory, 'test2.cpp'), 'w') as f:
            f.write(cpp_code)

    def test_get_c_and_cpp_files(self):
        c_files = get_c_and_cpp_files(self.test_directory)
        expected_c_files = [os.path.join(self.test_directory, 'test1.c'), os.path.join(self.test_directory, 'test2.cpp')]
        self.assertCountEqual(c_files, expected_c_files)

    def test_read_c_file(self):
        c_file_path = os.path.join(self.test_directory, 'test1.c')
        c_code = read_c_file(c_file_path)
        expected_c_code = '''
        #include <stdio.h>

        int main() {
            printf("Hello World!");
            return 0;
        }
        '''
        self.assertEqual(c_code, expected_c_code)

    def test_tokenize_c_function(self):
        c_code = '''
        #include <stdio.h>

        int add(int a, int b) {
            return a + b;
        }

        int main() {
            // add two numbers 
            int result = add(3, 4);
            printf("Result: %d", result);
            return 0;
        }
        '''
        functions = tokenize_c_function(c_code)
        expected_functions = [{'function': '        int add(int a, int b) {\n            return a + b;\n        }',
                                'mapper': {1: 4, 2: 5, 3: 6}},
                                {'function': '        int main() {\n            int result = add(3, 4);\n            printf("Result: %d", result);\n            return 0;\n        }',
                                'mapper': {1: 8, 2: 10, 3: 11, 4: 12, 5: 13}}
                            ]

        self.assertEqual(functions, expected_functions)


class FunctionStartTestCase(unittest.TestCase):
    def test_valid_function_start(self):
        valid_lines = [
            "int add(int a, int b) {",
            "void printMessage(const char* message) {",
            "int* allocateArray(size_t size) {",
            "struct Vector normalize(Vector v) {",
            "void func() {",
            "struct MyStruct* createStruct() {",
            "void doSomething(int x, int y, float z) {",
            "void foo(int x, int y = 10) {",
            "int bar(int x = 5, int y = 10) {",
            "char* processArgs(int argc, char *argv[]) {",
            "char* processArgs(int argc, int x=0 ,bool flag = false) {"
            "char* processArgs(int argc, char *argv, bool flag = false)",
            "int add(int a, int b) { return a + b; }",
        ]

        for line in valid_lines:
            self.assertTrue(is_start_of_function(line), f"Failed for line: {line}")


    def test_invalid_function_start(self):
        invalid_lines = [
            "int x = 10;",
            "int x = 10, y = 20;",
            "char *str = \"Hello World!\";",
            "char *str = \"Hello World!\"; int x = 10;",
        ]

        for line in invalid_lines:
            self.assertFalse(is_start_of_function(line),f"Failed for line: {line}")

if __name__ == '__main__':
    unittest.main()
