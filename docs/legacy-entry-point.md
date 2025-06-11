# Legacy Entry Point

Documentation for the backward compatibility features and legacy entry point in CSI-Predictor.

## What is the Legacy Entry Point?

The "Legacy Entry Point" refers to the `main.py` script that provides a **single, unified command-line interface** for running all project operations. It's called "legacy" because it represents an older, monolithic approach to structuring the project's entry points.

## How Does It Work?

The legacy entry point (`main.py`) works as follows:

### Single Command Interface

It provides one entry point with a `--mode` parameter to choose the operation:

```bash
python main.py --mode train    # Train a model
python main.py --mode eval     # Evaluate a model  
python main.py --mode both     # Train then evaluate
python main.py --mode hyperopt # Hyperparameter optimization
python main.py --mode sweep    # Initialize WandB sweep
python main.py --mode sweep-agent # Run WandB sweep agent
python main.py --mode train-optimized # Train with optimized hyperparameters
```

### Unified Arguments

All operations share the same argument parsing system:

```bash
python main.py --mode train --config custom_config.ini
python main.py --mode eval --wandb-project my-project
python main.py --mode hyperopt --n-trials 100
```

### Backward Compatibility

The legacy entry point ensures that:
- Old scripts and documentation continue to work
- Training pipelines don't break during updates  
- Users can migrate gradually to the new modular system

## Modern vs Legacy Entry Points

### Legacy Entry Point (main.py)

```bash
# Single file handles all operations
python main.py --mode train
python main.py --mode eval
python main.py --mode hyperopt
```

**Characteristics:**
- ✅ Simple single command
- ✅ Backward compatible
- ❌ Monolithic design
- ❌ All functionality in one file
- ❌ Harder to maintain and extend

### Modern Entry Points (Recommended)

```bash
# Modular, focused entry points
python -m src.train
python -m src.evaluate
python -m src.hyperopt  # If available
```

**Characteristics:**
- ✅ Modular design
- ✅ Each module has specific purpose
- ✅ Easier to maintain and test
- ✅ Better separation of concerns
- ✅ More Pythonic approach

## Why Do We Need Backward Compatibility?

### 1. **Existing Workflows**

Many users and automated systems may already be using the legacy interface:

```bash
# Existing scripts that need to keep working
#!/bin/bash
python main.py --mode train --config production.ini
python main.py --mode eval
```

### 2. **Documentation and Tutorials**

Existing documentation, tutorials, and blog posts reference the legacy interface. Maintaining compatibility prevents breaking these resources.

### 3. **Gradual Migration**

Users can migrate to the new system at their own pace:

```bash
# Old way (still works)
python main.py --mode train

# New way (recommended)
python -m src.train
```

### 4. **Enterprise Integration**

Enterprise environments often have long deployment cycles. Legacy compatibility ensures the system works in these environments while they plan migrations.

## Migration Guide

### From Legacy to Modern

**Old:**
```bash
python main.py --mode train --config my_config.ini
```

**New:**
```bash
python -m src.train --ini my_config.ini
```

**Old:**
```bash
python main.py --mode eval
```

**New:**
```bash
python -m src.evaluate
```

### Feature Mapping

| Legacy Command | Modern Equivalent | Notes |
|----------------|------------------|-------|
| `--mode train` | `python -m src.train` | Direct mapping |
| `--mode eval` | `python -m src.evaluate` | Direct mapping |
| `--mode both` | Run both commands separately | More explicit |
| `--mode hyperopt` | `python -m src.hyperopt` | If available |
| `--config file.ini` | `--ini file.ini` | Argument name change |

## Implementation Details

The legacy entry point is implemented as a dispatcher that routes to the appropriate modern modules:

```python
# Simplified structure of main.py
def main():
    args = parse_arguments()
    
    if args.mode == 'train':
        from src.train import main as train_main
        train_main(args)
    elif args.mode == 'eval':
        from src.evaluate import main as eval_main
        eval_main(args)
    elif args.mode == 'both':
        # Run training then evaluation
        from src.train import main as train_main
        from src.evaluate import main as eval_main
        train_main(args)
        eval_main(args)
    # ... other modes
```

## Best Practices

### 1. **Prefer Modern Entry Points**

For new code and documentation, use the modern entry points:

```bash
# Recommended
python -m src.train
python -m src.evaluate
```

### 2. **Maintain Legacy Documentation**

Keep documentation for both approaches until legacy support is officially deprecated:

```markdown
# Training a Model

## Modern Approach (Recommended)
```bash
python -m src.train
```

## Legacy Approach
```bash
python main.py --mode train
```

### 3. **Plan Migration Timeline**

- **Phase 1**: Both systems work (current state)
- **Phase 2**: Modern system is primary, legacy documented as deprecated
- **Phase 3**: Legacy system removed (major version update)

### 4. **Test Both Interfaces**

Ensure that both entry points produce equivalent results:

```bash
# Test equivalence
python main.py --mode train --config test.ini
python -m src.train --ini test.ini
# Results should be identical
```

## Deprecation Strategy

### Current Status

- **Legacy**: ✅ Fully supported
- **Modern**: ✅ Recommended for new usage
- **Documentation**: Shows both approaches

### Future Plans

1. **V2.0**: Legacy marked as deprecated but still functional
2. **V3.0**: Legacy removed, only modern entry points supported

### Migration Assistance

The project provides tools to help migration:

```bash
# Check usage patterns
grep -r "main.py --mode" scripts/  # Find legacy usage

# Automated conversion (if available)
python scripts/convert_legacy_scripts.py
```

## Example Usage Scenarios

### Development Workflow

```bash
# Quick testing with legacy (familiar)
python main.py --mode train --config dev.ini

# Production deployment with modern (recommended)
python -m src.train --ini production.ini
```

### CI/CD Integration

```bash
# Legacy CI script (still works)
#!/bin/bash
python main.py --mode train
if [ $? -eq 0 ]; then
    python main.py --mode eval
fi

# Modern CI script (recommended)
#!/bin/bash
python -m src.train
if [ $? -eq 0 ]; then
    python -m src.evaluate
fi
```

### Batch Processing

```bash
# Legacy batch processing
for config in configs/*.ini; do
    python main.py --mode train --config "$config"
done

# Modern batch processing
for config in configs/*.ini; do
    python -m src.train --ini "$config"
done
```

The legacy entry point ensures smooth transitions while encouraging adoption of better architectural patterns. This approach balances backward compatibility with forward progress. 