import pytest
import inspect
import sys
import os
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

def pytest_addoption(parser):
    parser.addoption(
        "--mode", action="store", default="startup", help="run tests against startup or solution implementation"
    )

def pytest_configure(config):
    mode = config.getoption("--mode")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_dir, mode)
    
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        pytest.exit(f"Mode directory not found: {target_dir}")
        
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    print(f"Running tests in {mode} mode, added {target_dir} to sys.path")

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to add student's source code to the test report on failure.
    """
    outcome = yield
    rep = outcome.get_result()
    
    # Only process if test failed during call phase
    if rep.when == "call" and rep.failed:
        # 1. Identify the student's object
        # We assume the test uses a fixture named 'layer' which contains the model layer instance
        layer = item.funcargs.get('layer', None)
        
        if layer is None:
            return

        # 2. Identify the method being tested based on test function name
        method_name = None
        test_name = item.name
        
        # Mapping rules based on test_bench.py naming convention
        if any(k in test_name for k in ["forward", "output", "stability", "loss"]):
            method_name = "__call__"
        elif any(k in test_name for k in ["backward", "grad", "parameter"]):
            method_name = "backpropagation"
            
        if method_name:
            try:
                # Get the method object
                method = getattr(layer, method_name, None)
                if method:
                    # 3. Get source code and location
                    source_lines, start_line = inspect.getsourcelines(method)
                    source_code = "".join(source_lines)
                    file_path = inspect.getsourcefile(method)
                    
                    # 4. Highlight code
                    # We use force_terminal=True to generate ANSI codes for console output
                    console = Console(force_terminal=True, width=100)
                    
                    # Create a syntax object
                    syntax = Syntax(
                        source_code, 
                        "python", 
                        theme="monokai", 
                        line_numbers=True,
                        start_line=start_line
                    )
                    
                    # Create a nice panel
                    title = f"Student Code: {layer.__class__.__name__}.{method_name}"
                    subtitle = f"{file_path}:{start_line}"
                    panel = Panel(
                        syntax, 
                        title=title, 
                        subtitle=subtitle,
                        border_style="red",
                        padding=(1, 2)
                    )
                    
                    with console.capture() as capture:
                        console.print(panel)
                    
                    formatted_output = capture.get()
                    
                    # 5. Add to report sections
                    # This ensures it shows up in console and logs
                    # For HTML reports, pytest-html usually handles ANSI codes if configured,
                    # or displays them as text.
                    rep.sections.append(("Student Implementation (Source)", formatted_output))
                    
            except Exception as e:
                # Fail gracefully if source code cannot be retrieved
                pass
