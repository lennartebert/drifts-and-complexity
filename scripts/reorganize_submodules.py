#!/usr/bin/env python3
"""Script to reorganize git submodules into a clean plugins structure."""

import subprocess
import shutil
from pathlib import Path
import sys
from typing import Dict, List, Tuple


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        print(f"Error: {e.stderr}")
        return False


def get_submodule_info() -> Dict[str, str]:
    """Get current submodule information."""
    try:
        result = subprocess.run(
            ["git", "submodule", "status"], 
            capture_output=True, text=True, check=True
        )
        
        submodules = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    commit_hash = parts[0]
                    path = parts[1]
                    submodules[path] = commit_hash
        
        return submodules
    except subprocess.CalledProcessError:
        print("Warning: Could not get submodule information")
        return {}


def reorganize_submodules() -> bool:
    """Reorganize submodules into plugins directory."""
    
    # Define the reorganization mapping
    submodule_mapping = {
        "process-complexity": "plugins/complexity",
        "cdrift-evaluation": "plugins/drift_detection", 
        "concept-drift-characterization": "plugins/drift_characterization"
    }
    
    print("🔍 Checking current submodule status...")
    current_submodules = get_submodule_info()
    
    if not current_submodules:
        print("❌ No submodules found. Make sure you're in a git repository with submodules.")
        return False
    
    print(f"Found submodules: {list(current_submodules.keys())}")
    
    # Create plugins directory
    plugins_dir = Path("plugins")
    plugins_dir.mkdir(exist_ok=True)
    
    print("\n📁 Creating plugins directory structure...")
    
    # Create README for plugins directory
    plugins_readme = plugins_dir / "README.md"
    plugins_readme.write_text("""# Plugins

This directory contains external dependencies and plugins for the drifts-and-complexity project.

## Plugin Structure

- `complexity/` - Process complexity metrics and analysis
- `drift_detection/` - Concept drift detection algorithms and evaluation
- `drift_characterization/` - Drift characterization and analysis tools

## Adding New Plugins

To add a new plugin:

1. Add the submodule to the plugins directory:
   ```bash
   git submodule add <repository-url> plugins/<plugin-name>
   ```

2. Update this README with the new plugin information

3. Update the main project documentation

## Plugin Dependencies

Each plugin may have its own dependencies. Check the individual plugin README files for specific requirements.
""")
    
    print("✅ Created plugins/README.md")
    
    # Process each submodule
    success = True
    for old_path, new_path in submodule_mapping.items():
        old_path_obj = Path(old_path)
        new_path_obj = Path(new_path)
        
        if not old_path_obj.exists():
            print(f"⚠️  Warning: {old_path} does not exist, skipping...")
            continue
            
        print(f"\n🔄 Moving {old_path} → {new_path}")
        
        # Create parent directory
        new_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the directory
        try:
            if new_path_obj.exists():
                shutil.rmtree(new_path_obj)
            shutil.move(str(old_path_obj), str(new_path_obj))
            print(f"✅ Moved {old_path} to {new_path}")
        except Exception as e:
            print(f"❌ Failed to move {old_path}: {e}")
            success = False
            continue
        
        # Update .gitmodules if it exists
        gitmodules_path = Path(".gitmodules")
        if gitmodules_path.exists():
            print(f"📝 Updating .gitmodules for {old_path} → {new_path}")
            try:
                content = gitmodules_path.read_text()
                content = content.replace(f'path = {old_path}', f'path = {new_path}')
                gitmodules_path.write_text(content)
                print(f"✅ Updated .gitmodules")
            except Exception as e:
                print(f"⚠️  Warning: Could not update .gitmodules: {e}")
    
    return success


def update_git_submodules() -> bool:
    """Update git submodule configuration after reorganization."""
    print("\n🔄 Updating git submodule configuration...")
    
    # Remove old submodule entries
    commands = [
        ["git", "rm", "--cached", "process-complexity"],
        ["git", "rm", "--cached", "cdrift-evaluation"], 
        ["git", "rm", "--cached", "concept-drift-characterization"],
    ]
    
    for cmd in commands:
        run_command(cmd, f"Removing {cmd[-1]} from git index")
    
    # Add new submodule entries
    new_commands = [
        ["git", "add", "plugins/complexity"],
        ["git", "add", "plugins/drift_detection"],
        ["git", "add", "plugins/drift_characterization"],
    ]
    
    for cmd in new_commands:
        run_command(cmd, f"Adding {cmd[-1]} to git index")
    
    return True


def update_imports() -> bool:
    """Update import statements to reflect new structure."""
    print("\n🔍 Looking for files that need import updates...")
    
    # Files that likely need import updates
    files_to_check = [
        "utils/complexity/metrics_adapters/vidgof_metrics_adapter.py",
        "assess_datasets.py",
        "combine_results.py",
        "run_bias_problematization.py",
        "run_bias_study.py",
    ]
    
    import_mappings = {
        "process-complexity": "plugins.complexity",
        "cdrift-evaluation": "plugins.drift_detection",
        "concept-drift-characterization": "plugins.drift_characterization",
    }
    
    updated_files = []
    
    for file_path in files_to_check:
        path_obj = Path(file_path)
        if not path_obj.exists():
            continue
            
        print(f"📝 Checking {file_path}...")
        content = path_obj.read_text()
        original_content = content
        
        for old_import, new_import in import_mappings.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                print(f"  Updated import: {old_import} → {new_import}")
        
        if content != original_content:
            path_obj.write_text(content)
            updated_files.append(file_path)
            print(f"✅ Updated {file_path}")
    
    if updated_files:
        print(f"\n✅ Updated {len(updated_files)} files with new import paths")
    else:
        print("\n✅ No files needed import updates")
    
    return True


def create_migration_guide() -> bool:
    """Create a migration guide for the reorganization."""
    migration_guide = Path("docs/SUBMODULE_MIGRATION.md")
    migration_guide.parent.mkdir(exist_ok=True)
    
    content = """# Submodule Migration Guide

This document describes the migration of git submodules to a cleaner plugins structure.

## Changes Made

### Directory Structure
- `process-complexity/` → `plugins/complexity/`
- `cdrift-evaluation/` → `plugins/drift_detection/`
- `concept-drift-characterization/` → `plugins/drift_characterization/`

### Benefits
1. **Clean naming**: No hyphens in directory names
2. **Logical grouping**: All external dependencies in one place
3. **Professional appearance**: Organized, clean structure
4. **Easy maintenance**: All submodules in one location

## Updated Imports

If you have code that imports from these submodules, update your imports:

```python
# Old imports
from process-complexity import Complexity
from cdrift-evaluation import evaluate
from concept-drift-characterization import characterize

# New imports  
from plugins.complexity import Complexity
from plugins.drift_detection import evaluate
from plugins.drift_characterization import characterize
```

## Git Submodule Commands

After migration, use these commands to work with submodules:

```bash
# Initialize all submodules
git submodule update --init --recursive

# Update a specific submodule
git submodule update --remote plugins/complexity

# Update all submodules
git submodule update --remote --recursive
```

## Verification

To verify the migration was successful:

1. Check that all plugin directories exist:
   ```bash
   ls -la plugins/
   ```

2. Verify submodule status:
   ```bash
   git submodule status
   ```

3. Test imports in your code:
   ```python
   from plugins.complexity import Complexity
   ```

## Rollback

If you need to rollback the changes:

1. Move directories back:
   ```bash
   mv plugins/complexity process-complexity
   mv plugins/drift_detection cdrift-evaluation
   mv plugins/drift_characterization concept-drift-characterization
   ```

2. Update .gitmodules file
3. Update import statements back to original paths
"""
    
    migration_guide.write_text(content)
    print(f"✅ Created migration guide: {migration_guide}")
    return True


def main():
    """Main migration function."""
    print("🚀 Starting submodule reorganization...")
    print("=" * 50)
    
    # Check if we're in a git repository
    if not Path(".git").exists():
        print("❌ Not in a git repository. Please run this from the project root.")
        return 1
    
    # Step 1: Reorganize submodules
    if not reorganize_submodules():
        print("❌ Failed to reorganize submodules")
        return 1
    
    # Step 2: Update git configuration
    if not update_git_submodules():
        print("❌ Failed to update git submodule configuration")
        return 1
    
    # Step 3: Update imports
    if not update_imports():
        print("❌ Failed to update imports")
        return 1
    
    # Step 4: Create migration guide
    if not create_migration_guide():
        print("❌ Failed to create migration guide")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 Submodule reorganization completed successfully!")
    print("\nNext steps:")
    print("1. Review the changes: git status")
    print("2. Test your code with the new import paths")
    print("3. Commit the changes: git add . && git commit -m 'Reorganize submodules into plugins structure'")
    print("4. Update any CI/CD configurations that reference the old paths")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
