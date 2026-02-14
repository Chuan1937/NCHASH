# NCHASH Documentation

Python implementation of HASH for earthquake focal mechanism determination.

## Contents

| File | Description |
|------|-------------|
| [API Reference](api.md) | Modules, functions, parameters |
| [Usage Guide](usage.md) | Examples and tutorials |

## Quick Links

- [Installation](usage.md#installation)
- [Basic Usage](usage.md#basic-usage)
- [With S/P Amplitude](usage.md#with-sp-amplitude)
- [run_hash()](api.md#run_hash)
- [run_hash_with_amp()](api.md#run_hash_with_amp)

## Performance

| Metric | Python | Fortran | Speedup |
|--------|--------|---------|---------|
| 24 events | 0.068s | 0.473s | **6.9x** |
| Per event | 2.85ms | 19.7ms | **6.9x** |

## Accuracy

- Dip error: < 10°
- Rake error: < 15°
- Algorithm: Exact match with Fortran

## References

- Hardebeck & Shearer (2002). BSSA, 92, 2264-2276
- Hardebeck & Shearer (2003). BSSA, 93, 2434-2444
