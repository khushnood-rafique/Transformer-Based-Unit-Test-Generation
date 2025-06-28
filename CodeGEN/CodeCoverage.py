import time
import coverage
import csv
import ast,re
from importlib import import_module
import os,sys
import pandas as pd
import ast,black
import textwrap
from importlib import import_module
import pytest

# Load the CSV file containing functions and unit tests
file_path = 'results/predicted_unit_tests.csv'
data = pd.read_csv(file_path)

# Directory to store generated scripts
generated_dir = "results/generated_scripts"
os.makedirs(generated_dir, exist_ok=True)

# File to combine all valid functions and tests
test_file = os.path.join(generated_dir, "test_suite.py")
csv_file = os.path.join(generated_dir, "test_execution_metrics.csv")

# Create CSV file for logging execution performance
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Test Name", "Execution Time (s)", "CPU Usage (%)", "Memory Usage (MB)", "GPU Utilization (%)"])
        
        
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

def fix_syntax(code):
    """
    Fixes common syntax issues in Python code.
    """
    # Replace inline semicolons with newlines
    code = code.replace(";", "\n")

    # Ensure colons are at the end of function headers and loops
    code = re.sub(r"(\bdef\b.*?\))\s*(?!:)", r"\1:", code)
    code = re.sub(r"(\bif\b.*?\))\s*(?!:)", r"\1:", code)
    code = re.sub(r"(\bfor\b.*?\))\s*(?!:)", r"\1:", code)
    code = re.sub(r"(\bwhile\b.*?\))\s*(?!:)", r"\1:", code)

    # Add proper indentation (basic level fix)
    code = "\n".join([line.lstrip() for line in code.split("\n")])

    return code

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
    f.write("import time\nimport pytest\nimport psutil\nimport subprocess\nimport csv\n\n")

    # Create a pytest fixture to measure resource usage
    f.write("""                 
@pytest.fixture(autouse=True)
def measure_resources(request):
    \"\"\"Measure execution time, CPU, memory, and GPU utilization for each test.\"\"\"
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=0.1)
    mem_start = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
    
        
    yield  # Run the actual test
    
    elapsed_time = time.time() - start_time
    cpu_end = psutil.cpu_percent(interval=0.1)
    mem_end = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert to MB
    
    try:
        gpu_usage = subprocess.getoutput("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    except:
        gpu_usage = "N/A"
    
    # Write results to CSV
    with open(\"""" + csv_file + """", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([request.node.name, elapsed_time, max(0, cpu_end - cpu_start), max(0, mem_end - mem_start), gpu_usage])
    
    # Print the results
    print(f"\\nTest: {request.node.name}")
    print(f"Execution Time: {elapsed_time:.5f} seconds")
    print(f"CPU Usage: {max(0, cpu_end - cpu_start):.2f}%")
    print(f"Memory Usage: {max(0, mem_end - mem_start):.2f} MB")
    print(f"GPU Utilization: {gpu_usage}%")
""")

    for index, row in data.iterrows():
        function_code = row['python_function']
        unit_test_code = row['predicted_unit_test'] 
        
        # Reformat function and unit test code
        #function_code = fix_syntax(function_code)
        function_code = reformat_code(function_code)
        unit_test_code = format_predicted_unit_test(unit_test_code)
        
        # Write each function and its test to the combined script
        if(function_code):
            if unit_test_code and is_valid_python_code(unit_test_code) and has_valid_assertions(unit_test_code): 
                f.write(f"# Function {index + 1}\n")
                f.write(function_code + "\n\n")
                f.write(f"# Unit Test {index + 1}\n")
                f.write(unit_test_code + "\n\n")
            else:
                print(f"Skipping Test {index + 1}: Invalid syntax or No valid assertions found.")
        else:
            print(f"Skipping Function {index + 1}: Invalid syntax detected.")

# Perform coverage analysis and execution
cov = coverage.Coverage()
cov.start()
sys.path.insert(0, generated_dir)

# Extract the module name from the test file
module_name = os.path.splitext(os.path.basename(test_file))[0]

# # Dynamically import the module
try:
    test_module = import_module(module_name)
    print(f"Successfully imported module: {module_name}")
    
    # Run the test suite using pytest
    print("\nRunning tests with pytest...")
    pytest.main([test_file, "--disable-warnings"])

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
