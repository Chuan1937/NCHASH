# Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, NumPy, Numba, SciPy

---

## Basic Usage

Determine focal mechanism from P-wave polarities:

```python
from nchash import run_hash
import numpy as np

# Input data
p_azi = np.array([45, 135, 225, 315, 0, 90, 180, 270])      # Azimuths
p_the = np.array([30, 45, 60, 75, 40, 50, 55, 65])           # Takeoff angles
p_pol = np.array([1, -1, 1, -1, 1, -1, 1, -1])               # Polarities
p_qual = np.array([0, 0, 0, 0, 0, 0, 0, 0])                  # Quality

# Run
result = run_hash(p_azi, p_the, p_pol, p_qual)

# Output
print(f"Strike: {result['strike_avg']:.1f}°")
print(f"Dip: {result['dip_avg']:.1f}°")
print(f"Rake: {result['rake_avg']:.1f}°")
print(f"Quality: {result['quality']}")
```

---

## With S/P Amplitude

Add S/P amplitude ratios for better constraint:

```python
from nchash import run_hash_with_amp

# S/P ratios in log10 scale
# 0.0 = no data, typical range: -1.0 to 2.0
sp_amp = np.array([0.3, -0.2, 0.5, 0.0, 0.4, -0.1, 0.6, 0.2])

result = run_hash_with_amp(p_azi, p_the, p_pol, sp_amp)

print(f"Polarity misfit: {result['mfrac']*100:.1f}%")
print(f"Amplitude misfit: {result['mavg']:.2f}")
```

---

## From Input File

Process multiple events from HASH input file:

```python
from nchash import run_hash_from_file

results = run_hash_from_file("example1.inp")

for result in results:
    if result['success']:
        print(f"S={result['strike_avg']:.0f} D={result['dip_avg']:.0f} "
              f"R={result['rake_avg']:.0f} Q={result['quality']}")
```

---

## Monte Carlo Uncertainty

The algorithm automatically performs Monte Carlo perturbation when `nmc > 1`:

```python
# 30 Monte Carlo trials (default)
result = run_hash(p_azi, p_the, p_pol, p_qual, nmc=30)

# More trials = more stable uncertainty estimate
result = run_hash(p_azi, p_the, p_pol, p_qual, nmc=100)
```

---

## Grid Resolution

Adjust grid search resolution:

```python
# Coarse (fast)
result = run_hash(..., dang=10)

# Medium (default)
result = run_hash(..., dang=5)

# Fine (slow, more accurate)
result = run_hash(..., dang=4)
```

---

## Acceptance Criteria

Control which mechanisms are accepted:

```python
# Allow 15% bad polarities (default: 10%)
result = run_hash(..., badfrac=0.15)

# Minimum 6 polarities (default: 8)
result = run_hash(..., npolmin=6)

# Max azimuth gap 120° (default: 90°)
result = run_hash(..., max_agap=120)
```

---

## Low-Level Access

Access individual acceptable mechanisms:

```python
from nchash import focalmc

# Prepare MC arrays
nmc = 30
p_azi_mc = np.zeros((nsta, nmc))
p_the_mc = np.zeros((nsta, nmc))
p_azi_mc[:, 0] = p_azi
p_the_mc[:, 0] = p_the
# ... add perturbations ...

# Run core algorithm
result = focalmc(p_azi_mc, p_the_mc, p_pol, p_qual,
                 nsta, nmc, dang=5.0, maxout=500, nextra=2, ntotal=5)

# All acceptable mechanisms
for i in range(result['nf']):
    print(f"Solution {i+1}: S={result['strike'][i]:.1f} "
          f"D={result['dip'][i]:.1f} R={result['rake'][i]:.1f}")
```

---

## Calculate Misfit

Check misfit for a known mechanism:

```python
from nchash import get_misfit

mfrac, stdr = get_misfit(nsta, p_azi, p_the, p_pol, p_qual,
                          strike=45, dip=60, rake=-90)
print(f"Misfit: {mfrac*100:.1f}%")
```

---

## Read Phase Files

Parse HASH phase files:

```python
from nchash.io import read_phase_file

events = read_phase_file("north1.phase")

for event in events:
    print(f"Event {event['id']}: {len(event['stations'])} stations")
    for sta in event['stations']:
        print(f"  {sta['name']}: pol={sta['polarity']}")
```

---

## Input File Format

Create a HASH input file (`example.inp`):

```
scsn.pol           # Polarity reversal file
north1.phase       # Phase data
output.out         # Output file 1
out2.out           # Output file 2
8                  # Minimum polarities
90                 # Max azimuth gap
60                 # Max plunge gap
5                  # Grid angle
30                 # Monte Carlo trials
500                # Max output
0.1                # Bad fraction
120                # Max distance
45                 # C-angle
0.1                # Prob max
```

---

## Common Workflows

### Single Event Analysis

```python
from nchash import run_hash
import numpy as np

# Your data
p_azi = np.array([...])
p_the = np.array([...])
p_pol = np.array([...])
p_qual = np.zeros_like(p_pol)

result = run_hash(p_azi, p_the, p_pol, p_qual, dang=5, nmc=30)

if result['success']:
    print(f"Strike: {result['strike_avg']:.1f}°")
    print(f"Dip: {result['dip_avg']:.1f}°")
    print(f"Rake: {result['rake_avg']:.1f}°")
    print(f"Quality: {result['quality']}")
    print(f"Solutions: {result['nout2']}")
else:
    print(f"Failed: {result['quality']}")
```

### Batch Processing

```python
from nchash import run_hash_from_file

results = run_hash_from_file("events.inp")

# Count by quality
quality_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0}
for r in results:
    quality_count[r['quality']] += 1

print(f"A: {quality_count['A']}, B: {quality_count['B']}, "
      f"C: {quality_count['C']}, D: {quality_count['D']}, "
      f"F: {quality_count['F']}")
```

---

## Tips

1. **Minimum 8 polarities** recommended for reliable solutions
2. **Azimuth gap < 90°** for well-constrained mechanisms
3. **Use S/P ratios** when available for better constraint
4. **nmc=30** is usually sufficient for uncertainty estimation
5. **dang=5°** balances speed and accuracy
