import time
import coverage
import ast
from importlib import import_module
import os,sys
import pandas as pd
import ast,black
import textwrap
from importlib import import_module

# Load the CSV file containing functions and unit tests
file_path = 'results/predicted_unit_tests.csv'
data = pd.read_csv(file_path)

# Directory to store generated scripts
generated_dir = "results/generated_scripts"
os.makedirs(generated_dir, exist_ok=True)

# File to combine all valid functions and tests
test_file = os.path.join(generated_dir, "test_suite.py")

def format_predicted_unit_test(predicted):
    
    # Split the test case by 'assert' to separate the individual assertions
    parts = predicted.strip().split("assert")
    
    # Remove the first part, which contains the function definition
    function_line = parts[0].strip()
    
    # Rebuild the test case with proper indentation
    indented_test_case = [f"{function_line}"]
    for part in parts[1:]:
        indented_test_case.append(f"    assert {part.strip()}")
    
    # Return the formatted test case
    return "\n".join(indented_test_case)
def reformat_code(code):
    """
    Reformat and normalize the indentation of Python code.
    Convert multiline strings to executable Python code.
    """
    try:
        # Dedent and strip leading/trailing spaces
        code = textwrap.dedent(code).strip()
        # Parse the code to validate syntax
        code = ast.parse(code)
        formatted_code = ast.unparse(code)
        # Use black to format the code properly
        formatted_code = black.format_str(formatted_code, mode=black.Mode())
        return formatted_code
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return False


def is_valid_python_code(code):
    """Check if the given Python code is syntactically valid."""
    try:
    
        # Validate syntax
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"Syntax Error: {e}")
        return False

def has_valid_assertions(code):
    """Check if the code contains at least one valid 'assert' statement."""
    try:
    
        # Validate syntax
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                return True
        return False
    except SyntaxError as e:
        print(f"Syntax Error while parsing for assertions: {e}")
        return False

with open(test_file, "w") as f:
    f.write("import time\nimport pytest\n\n")

    for index, row in data.iterrows():
        function_code = row['python_function']
        unit_test_code = row['predicted_unit_test']
        
        #print("Code before reformat....")
        #print(f"Function {index + 1} Code:\n{function_code}\t")
        #print(f"Unit Test {index + 1} Code:\n{unit_test_code}\n")
        
        # Reformat function and unit test code
        function_code = reformat_code(function_code)
        unit_test_code = format_predicted_unit_test(unit_test_code)
        
        #print("Code after reformat....")
        print(f"Function {index + 1} Code:\n{function_code}\t")
        print(f"Unit Test {index + 1} Code:\n{unit_test_code}\n")
        
        # Validate function and unit test code
        # if function_code and is_valid_python_code(function_code):
        #     if unit_test_code and is_valid_python_code(unit_test_code) and has_valid_assertions(unit_test_code):
            
        # Write each function and its test to the combined script
        f.write(f"# Function {index + 1}\n")
        f.write(function_code + "\n\n")
        f.write(f"# Unit Test {index + 1}\n")

        # Wrap unit test with timing logic
        timed_test_code = (
            f"start_time = time.time()\n"
            f"{unit_test_code}\n"
            f"elapsed_time = time.time() - start_time\n"
            f"print(f'Test {index + 1} execution time: {{elapsed_time:.5f}} seconds')\n"
            )
        f.write(timed_test_code + "\n\n")
        #     else:
        #         print(f"Skipping Test {index + 1}: Invalid syntax or No valid assertions found.")
        # else:
        #     print(f"Skipping Function {index + 1}: Invalid syntax detected.")

# Perform coverage analysis and execution
cov = coverage.Coverage()
cov.start()
sys.path.insert(0, generated_dir)

# Extract the module name from the test file
module_name = os.path.splitext(os.path.basename(test_file))[0]

# Dynamically import the module
try:
    try:
        test_module = import_module(module_name)
        print(f"Successfully imported module: {module_name}")
    except Exception as e:
        print(f"Error importing module: {e}")
    
    # Explicitly call each test function (you could also use unittest or pytest for automated discovery)
    for name in dir(test_module):
        if name.startswith("test_"):  # test function names usually start with 'test_'
            test_func = getattr(test_module, name)
            if callable(test_func):  # Ensure it's a function
                try:
                    print(f"Running test function: {name}")
                    test_func()  # Call the test function
                except Exception as e:
                    print(f"Error in test function {name}: {e}")

except Exception as e:
    print(f"Error during test execution: {e}")

    
finally:
    cov.stop()
    cov.save()

# Generate and display coverage report
print("\nCode Coverage Report:")
cov.report()
html_report_dir = os.path.join(generated_dir, "htmlcov")
cov.html_report(directory=html_report_dir)
print(f"HTML report generated at {html_report_dir}")
