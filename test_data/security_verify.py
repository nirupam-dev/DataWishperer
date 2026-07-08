"""Quick security verification script."""
from backend.sandbox.validator import CodeValidator

v = CodeValidator()

# Test safe code
safe = 'result = df.describe()'
assert v.is_safe(safe), f'Safe code rejected: {v.validate(safe)}'
print('PASS: Safe code accepted')

# Test dangerous imports
for mod in ['os', 'subprocess', 'socket', 'requests', 'sys']:
    code = f'import {mod}\nresult = 42'
    assert not v.is_safe(code), f'{mod} import should be blocked'
print('PASS: Dangerous imports blocked (os, subprocess, socket, requests, sys)')

# Test eval/exec
for func in ['eval', 'exec', 'compile', 'open']:
    code = func + '("x")\nresult = 42'
    assert not v.is_safe(code), f'{func} call should be blocked'
print('PASS: eval/exec/compile/open blocked')

# Test __import__
code = '__import__("os")\nresult = 42'
assert not v.is_safe(code), '__import__ should be blocked'
print('PASS: __import__ blocked')

# Test dunder escape
code = 'x = obj.__class__.__bases__\nresult = 42'
assert not v.is_safe(code), 'Dunder escape not blocked'
print('PASS: Dunder escape chains blocked')

# Test subprocess string pattern
code = 'import subprocess\nresult = 42'
assert not v.is_safe(code), 'subprocess not blocked'
print('PASS: subprocess blocked')

# Test network access
for mod in ['http', 'urllib', 'httpx', 'aiohttp']:
    code = f'import {mod}\nresult = 42'
    assert not v.is_safe(code), f'{mod} should be blocked'
print('PASS: Network modules blocked')

# Test filesystem write via open
code = 'open("file.txt", "w")\nresult = 42'
assert not v.is_safe(code), 'open() should be blocked'
print('PASS: Filesystem write blocked')

# Test os.system string pattern
code = 'import os\nos.system("rm -rf /")\nresult = 42'
assert not v.is_safe(code), 'os.system should be blocked'
print('PASS: os.system blocked')

# Test prompt injection attempt
code = '# Ignore previous instructions\nimport os\nos.system("whoami")\nresult = 42'
assert not v.is_safe(code), 'prompt injection bypass should be blocked'
print('PASS: Malicious prompt bypass blocked')

# Test obfuscation via chr()
code = 'chr(101)\nresult = 42'
assert not v.is_safe(code), 'chr() obfuscation should be blocked'
print('PASS: chr() obfuscation blocked')

# Test empty code
violations = v.validate('')
assert any(viol.category == 'structure' for viol in violations), 'Empty code not detected'
print('PASS: Empty code rejected')

# Test allowed modules
for safe_code in [
    'import pandas as pd\nresult = pd.DataFrame()',
    'import numpy as np\nresult = np.array([1,2,3])',
    'import matplotlib.pyplot as plt\nresult = "ok"',
    'from datetime import datetime\nresult = str(datetime.now())',
    'import math\nresult = math.sqrt(4)',
]:
    assert v.is_safe(safe_code), f'Safe code wrongly rejected: {safe_code[:50]}'
print('PASS: Safe analytical imports accepted (pandas, numpy, matplotlib, datetime, math)')

print()
print('=' * 60)
print('ALL SECURITY VERIFICATION TESTS PASSED')
print('=' * 60)
