# Submodule Reorganization Guide

This guide helps you reorganize your git submodules into a clean, professional structure.

## 🎯 **Goal**

Transform this messy structure:
```
drifts-and-complexity/
├── process-complexity/           # ❌ Hyphen, scattered
├── cdrift-evaluation/            # ❌ Hyphen, scattered
├── concept-drift-characterization/ # ❌ Hyphen, scattered
└── utils/
```

Into this clean structure:
```
drifts-and-complexity/
├── plugins/                      # ✅ Clean, organized
│   ├── complexity/              # ✅ No hyphens
│   ├── drift_detection/         # ✅ Descriptive name
│   ├── drift_characterization/  # ✅ Descriptive name
│   └── README.md
├── utils/
└── ...
```

## 🚀 **Quick Migration (Automated)**

Run the automated migration script:

```bash
# Make the script executable
chmod +x scripts/reorganize_submodules.py

# Run the migration
python scripts/reorganize_submodules.py
```

## 📋 **Manual Migration Steps**

If you prefer to do it manually:

### 1. **Create plugins directory**
```bash
mkdir plugins
```

### 2. **Move submodules**
```bash
# Move each submodule
mv process-complexity plugins/vidgof_complexity
mv cdrift-evaluation plugins/cdrift_evaluation
mv concept-drift-characterization plugins/drift_characterization
```

### 3. **Update .gitmodules**
Edit `.gitmodules` to reflect new paths:
```ini
[submodule "plugins/complexity"]
	path = plugins/complexity
	url = <your-process-complexity-repo-url>

[submodule "plugins/drift_detection"]
	path = plugins/drift_detection
	url = <your-cdrift-evaluation-repo-url>

[submodule "plugins/drift_characterization"]
	path = plugins/drift_characterization
	url = <your-concept-drift-characterization-repo-url>
```

### 4. **Update git index**
```bash
# Remove old entries
git rm --cached process-complexity
git rm --cached cdrift-evaluation
git rm --cached concept-drift-characterization

# Add new entries
git add plugins/complexity
git add plugins/drift_detection
git add plugins/drift_characterization
```

### 5. **Update imports in your code**
Search and replace in your Python files:

| Old Import | New Import |
|------------|------------|
| `from process-complexity import` | `from plugins.vidgof_complexity import` |
| `from cdrift-evaluation import` | `from plugins.cdrift_evaluation import` |
| `from concept-drift-characterization import` | `from plugins.drift_characterization import` |

### 6. **Update any hardcoded paths**
Search for references to the old paths in:
- Configuration files
- Documentation
- CI/CD scripts
- Jupyter notebooks

## ✅ **Verification**

After migration, verify everything works:

```bash
# 1. Check directory structure
ls -la plugins/

# 2. Verify submodule status
git submodule status

# 3. Test imports
python -c "from plugins.complexity import Complexity; print('✅ Complexity import works')"
python -c "from plugins.drift_detection import evaluate; print('✅ Drift detection import works')"
python -c "from plugins.drift_characterization import characterize; print('✅ Drift characterization import works')"

# 4. Run your tests
pytest tests/
```

## 🔧 **Common Issues & Solutions**

### Issue: Import errors after migration
**Solution**: Update your Python path or add `plugins/` to `sys.path`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "plugins"))
```

### Issue: Git submodule not found
**Solution**: Reinitialize submodules:
```bash
git submodule update --init --recursive
```

### Issue: CI/CD fails after migration
**Solution**: Update your CI/CD configuration files to use new paths:
- `.github/workflows/ci.yml`
- `Dockerfile`
- Any deployment scripts

## 📚 **Benefits of This Structure**

1. **Professional appearance**: Clean, organized directory structure
2. **Easy maintenance**: All external dependencies in one place
3. **Clear separation**: Your code vs. external plugins
4. **Scalable**: Easy to add new plugins in the future
5. **No naming conflicts**: No hyphens or special characters

## 🎯 **Future Plugin Management**

To add new plugins in the future:

```bash
# Add a new plugin
git submodule add <repo-url> plugins/<plugin-name>

# Update a plugin
git submodule update --remote plugins/<plugin-name>

# Remove a plugin
git submodule deinit plugins/<plugin-name>
git rm plugins/<plugin-name>
```

## 📝 **Commit Message**

When you commit these changes:
```bash
git add .
git commit -m "refactor: reorganize submodules into plugins structure

- Move process-complexity → plugins/vidgof_complexity
- Move cdrift-evaluation → plugins/cdrift_evaluation
- Move concept-drift-characterization → plugins/drift_characterization
- Update imports and documentation
- Improve project organization and maintainability"
```

This reorganization will make your project much more professional and maintainable! 🎉
