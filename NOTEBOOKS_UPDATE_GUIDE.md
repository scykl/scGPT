# Updating Tutorials for Intel XPU Support

This document explains how to update the scGPT tutorials to support Intel XPU.

## Affected Notebooks

The following notebooks contain hardcoded CUDA device selection:
- `tutorials/Tutorial_Annotation.ipynb`
- `tutorials/Tutorial_GRN.ipynb`
- `tutorials/Tutorial_Attention_GRN.ipynb`
- `tutorials/Tutorial_Perturbation.ipynb`
- `tutorials/Tutorial_Integration.ipynb`
- `tutorials/Tutorial_Multiomics.ipynb`

## How to Update Each Notebook

### Step 1: Find the Device Selection Cell

Look for code like:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Step 2: Replace with New Code

Replace with:
```python
from scgpt.utils import get_device

device = get_device("auto")  # Auto-detects XPU > CUDA > CPU
print(f"Using device: {device}")
```

### Step 3: Find Autocast Usage (if any)

Some notebooks might have:
```python
with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
    # code
```

Replace with:
```python
from scgpt.utils import AutocastConfig

autocast_config = AutocastConfig(device, enabled=True)
with torch.no_grad(), autocast_config.autocast_context():
    # code
```

## Example Updates by Notebook

### Tutorial_Annotation.ipynb

**Original (Line ~1242)**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Updated**:
```python
from scgpt.utils import get_device
device = get_device("auto")
print(f"Using device: {device}")
```

**Also update** (Line ~1607):
```python
# Original
with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
    # ...

# Updated
from scgpt.utils import AutocastConfig
autocast_config = AutocastConfig(device, enabled=config.amp)
with torch.no_grad(), autocast_config.autocast_context():
    # ...
```

### Tutorial_GRN.ipynb

**Original (Line ~509)**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Updated**:
```python
from scgpt.utils import get_device
device = get_device("auto")
print(f"Using device: {device}")
```

### Tutorial_Integration.ipynb

**Original (Line ~667)**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Updated**:
```python
from scgpt.utils import get_device
device = get_device("auto")
print(f"Using device: {device}")
```

### Tutorial_Perturbation.ipynb

**Original (Line ~121)**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Updated**:
```python
from scgpt.utils import get_device
device = get_device("auto")
print(f"Using device: {device}")
```

### Tutorial_Multiomics.ipynb

**Original (Line ~871)**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Updated**:
```python
from scgpt.utils import get_device
device = get_device("auto")
print(f"Using device: {device}")
```

## Automated Update Script

Here's a Python script to automatically update notebooks:

```python
import json
import re

def update_notebook(notebook_path):
    """Update notebook to use new device utilities."""
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    updates_made = 0
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        
        # Pattern 1: Device selection
        pattern1 = r'device = torch\.device\("cuda" if torch\.cuda\.is_available\(\) else "cpu"\)'
        if re.search(pattern1, source):
            new_source = re.sub(pattern1, 
                '''from scgpt.utils import get_device

device = get_device("auto")  # Auto-detects XPU > CUDA > CPU
print(f"Using device: {device}")''', source)
            cell['source'] = new_source.split('\n')
            updates_made += 1
        
        # Pattern 2: Autocast with CUDA
        pattern2 = r'with torch\.no_grad\(\), torch\.cuda\.amp\.autocast\(enabled=([^)]+)\):'
        if re.search(pattern2, source):
            new_source = re.sub(pattern2,
                r'''from scgpt.utils import AutocastConfig
autocast_config = AutocastConfig(device, enabled=\1)
with torch.no_grad(), autocast_config.autocast_context():''', source)
            cell['source'] = new_source.split('\n')
            updates_made += 1
    
    if updates_made > 0:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Updated {notebook_path}: {updates_made} changes made")
    else:
        print(f"No changes needed for {notebook_path}")

# Usage
if __name__ == "__main__":
    notebooks = [
        "tutorials/Tutorial_Annotation.ipynb",
        "tutorials/Tutorial_GRN.ipynb",
        "tutorials/Tutorial_Attention_GRN.ipynb",
        "tutorials/Tutorial_Perturbation.ipynb",
        "tutorials/Tutorial_Integration.ipynb",
        "tutorials/Tutorial_Multiomics.ipynb",
    ]
    
    for nb_path in notebooks:
        update_notebook(nb_path)
```

## Manual Update Instructions

If you prefer to update manually via Jupyter notebook UI:

1. Open the notebook in Jupyter Lab/Notebook
2. Find the cell with device initialization (usually near beginning)
3. Add import: `from scgpt.utils import get_device`
4. Replace device line with: `device = get_device("auto")`
5. Find any `torch.cuda.amp.autocast` calls
6. Replace with `AutocastConfig(device, enabled=...).autocast_context()`
7. Save notebook

## Verification

After updating, verify that:

1. **Device Auto-Detection Works**:
   ```python
   from scgpt.utils import get_device
   device = get_device("auto")
   print(device)  # Should show device type
   ```

2. **Notebook Runs**:
   - Run the updated notebook
   - Check that model loads correctly
   - Verify training/inference works

3. **No Breaking Changes**:
   - Results should match original
   - Performance should be similar or better

## Testing Updated Notebooks

### Quick Test
```bash
# Test device detection in notebook
jupyter notebook tutorials/Tutorial_Integration.ipynb
# Run first few cells, verify device is detected
```

### Full Test
```bash
# Run full notebook
jupyter nbconvert --to notebook --execute tutorials/Tutorial_Integration.ipynb
```

## Benefits After Update

✅ Works on Intel Arc GPU (with IPEX)  
✅ Works on NVIDIA CUDA GPUs (unchanged)  
✅ Works on CPU (fallback)  
✅ Better code reusability  
✅ Example for users  
✅ Easier to maintain  

## Backward Compatibility

All updates are **100% backward compatible**:
- Old code still works
- New code uses new utilities
- No API breaking changes
- Can gradually transition

## Summary of Changes

For each notebook, typically need to make 1-3 changes:

1. Replace device initialization (1 location)
2. Replace autocast context if used (0-2 locations)
3. Add import statements (automatically handled by code)

**Estimated time per notebook**: 2-5 minutes

## Notes

- Changes are optional but recommended
- Users can still use old patterns
- New patterns are cleaner and more portable
- Documentation provided for all patterns

## Questions?

See the main documentation:
- `INTEL_XPU_SUPPORT.md` - Complete guide
- `QUICK_REFERENCE.md` - Quick examples
- `IMPLEMENTATION_SUMMARY.md` - Technical details
