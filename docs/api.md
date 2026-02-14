# API Reference

## Modules

| Module | Description |
|--------|-------------|
| `driver.py` | Main entry points |
| `core.py` | Grid search algorithm |
| `amp_subs.py` | S/P amplitude ratio |
| `uncertainty.py` | Uncertainty analysis |
| `io.py` | File I/O |
| `utils.py` | Coordinate conversions |
| `velocity.py` | Velocity model tables |

---

## driver.py

Main entry point functions.

### run_hash()

Determine focal mechanism from P-wave polarities.

```python
result = run_hash(p_azi, p_the, p_pol, p_qual, dang=5.0, nmc=30, ...)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `p_azi` | ndarray | required | Azimuths (degrees), shape (nsta,) or (nsta, nmc) |
| `p_the` | ndarray | required | Takeoff angles (degrees) |
| `p_pol` | ndarray | required | Polarities: 1=up, -1=down |
| `p_qual` | ndarray | required | Quality: 0=impulsive, 1=emergent |
| `dang` | float | 5.0 | Grid angle increment |
| `nmc` | int | 30 | Monte Carlo trials |
| `maxout` | int | 500 | Max output mechanisms |
| `badfrac` | float | 0.1 | Allowed bad polarity fraction |
| `npolmin` | int | 8 | Minimum polarities |

**Returns:**

```python
{
    'success': bool,       # Solution found
    'strike_avg': float,   # Strike (degrees)
    'dip_avg': float,      # Dip (degrees)
    'rake_avg': float,     # Rake (degrees)
    'quality': str,        # A, B, C, D, E, or F
    'mfrac': float,        # Misfit fraction (0-1)
    'prob': float,         # Solution probability
    'stdr': float,         # Station distribution ratio
    'nout2': int,          # Number of solutions
}
```

---

### run_hash_with_amp()

Determine focal mechanism using polarities + S/P amplitude ratios.

```python
result = run_hash_with_amp(p_azi, p_the, p_pol, sp_amp, dang=5.0, nmc=30, ...)
```

**Additional Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `sp_amp` | ndarray | S/P amplitude ratios (log10 scale), 0.0 = no data |

**Additional Returns:**

```python
{
    'mavg': float,   # Amplitude misfit (log10)
    'npol': int,     # Polarity count
    'nspr': int,     # S/P ratio count
}
```

---

### run_hash_from_file()

Process events from HASH input file.

```python
results = run_hash_from_file("example.inp")
```

---

## core.py

Core grid search algorithm.

### focalmc()

Find acceptable focal mechanisms via grid search.

```python
result = focalmc(p_azi_mc, p_the_mc, p_pol, p_qual, npol, nmc, dang, maxout, nextra, ntotal)
```

**Returns:**

```python
{
    'nf': int,              # Number of solutions
    'strike': ndarray,      # Strikes (nf,)
    'dip': ndarray,         # Dips (nf,)
    'rake': ndarray,        # Rakes (nf,)
    'faults': ndarray,      # Fault normals (3, nf)
    'slips': ndarray,       # Slip vectors (3, nf)
}
```

---

### get_misfit()

Calculate polarity misfit for a mechanism.

```python
mfrac, stdr = get_misfit(npol, p_azi, p_the, p_pol, p_qual, strike, dip, rake)
```

---

### get_gap()

Calculate azimuthal and plunge gaps.

```python
magap, mpgap = get_gap(npol, p_azi, p_the)
```

---

## amp_subs.py

S/P amplitude ratio functions.

### focalamp_mc()

Grid search with S/P amplitude constraint.

```python
result = focalamp_mc(p_azi_mc, p_the_mc, sp_amp, p_pol, npsta, nmc, dang, maxout, nextra, ntotal, qextra, qtotal)
```

---

### get_misf_amp()

Calculate both polarity and amplitude misfit.

```python
mfrac, mavg, stdr = get_misf_amp(npol, p_azi, p_the, sp_ratio, p_pol, strike, dip, rake)
```

---

## uncertainty.py

Uncertainty analysis functions.

### mech_prob()

Calculate mechanism probability and average.

```python
result = mech_prob(nf, norm1, norm2, cangle=45.0, prob_max=0.1)
```

**Returns:**

```python
{
    'nsltn': int,           # Number of solutions
    'strike_avg': ndarray,  # Average strikes
    'dip_avg': ndarray,     # Average dips
    'rake_avg': ndarray,    # Average rakes
    'prob': ndarray,        # Probabilities
    'rms_diff': ndarray,    # RMS differences (2, nsltn)
}
```

---

### mech_rot()

Calculate minimum rotation angle between mechanisms.

```python
angle = mech_rot(norm1, slip1, norm2, slip2)
```

---

### mech_avg()

Calculate average mechanism.

```python
norm1_avg, norm2_avg = mech_avg(nf, norm1, norm2)
```

---

## io.py

File I/O functions.

### read_phase_file()

Read HASH phase file.

```python
events = read_phase_file("north1.phase")
```

Supports formats 1-4 from HASH v1.2.

---

### read_station_file()

Read station coordinates.

```python
stations = read_station_file("scsn.stations")
```

---

### read_hash_input_file()

Read HASH driver input file.

```python
params = read_hash_input_file("example1.inp")
```

---

### write_mechanism_output()

Write mechanism results.

```python
write_mechanism_output("output.out", events, results)
```

---

## utils.py

Coordinate conversion utilities.

### fp_coord_angles_to_vectors()

Convert strike/dip/rake to fault normal and slip vectors.

```python
norm, slip = fp_coord_angles_to_vectors(strike, dip, rake)
```

---

### fp_coord_vectors_to_angles()

Convert fault normal and slip vectors to strike/dip/rake.

```python
strike, dip, rake = fp_coord_vectors_to_angles(norm, slip)
```

---

## velocity.py

Velocity model table generation.

### make_table()

Generate takeoff angle table from velocity model files.

```python
make_table(["vz.layer", "vz.half"])
```

---

### get_angle()

Interpolate takeoff angle from table.

```python
angle, iflag = get_angle(ip, del_dist, qdep)
```

---

## Quality Rating

| Grade | Criteria |
|-------|----------|
| **A** | prob > 0.8, var ≤ 25°, misfit ≤ 15% |
| **B** | prob > 0.6, var ≤ 35°, misfit ≤ 20% |
| **C** | prob > 0.5, var ≤ 45°, misfit ≤ 30% |
| **D** | Solution found, below C criteria |
| **E** | Gap too large |
| **F** | No solution |
