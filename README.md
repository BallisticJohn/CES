# Air Combat Simulator - F-86 Sabre vs MiG-15

A highly accurate, physically realistic air combat simulator focusing on Korean War-era jet aircraft, designed for Monte Carlo analysis and empirical validation.

## Features

- **Physically accurate flight dynamics**: 6-DOF (Degrees of Freedom) equations of motion
- **Realistic aerodynamics**: Lift, drag, thrust modeling based on empirical data
- **Accurate aircraft models**: F-86 Sabre and MiG-15 with historical specifications
- **Monte Carlo simulation**: Run thousands of engagement scenarios
- **Weapon systems**: Period-accurate gun systems (M3 Browning .50 cal, NR-23 23mm)
- **Damage modeling**: Component-based damage system
- **Historical validation**: Uses documented performance data from Korean War
- **Visualization**: 3D flight paths, engagement analysis, statistical outputs

## Aircraft Specifications

### F-86F Sabre
- **Engine**: General Electric J47-GE-27 (5,910 lbf / 26.3 kN thrust)
- **Maximum speed**: 687 mph (1,105 km/h, Mach 0.9) at sea level
- **Service ceiling**: 49,000 ft (14,935 m)
- **Rate of climb**: 9,300 ft/min (47.2 m/s)
- **Wing loading**: 52.5 lb/ft² (256 kg/m²)
- **Thrust/weight**: 0.38
- **Armament**: 6× .50 cal (12.7mm) M3 Browning machine guns (1,800 rounds total)
- **Empty weight**: 10,950 lb (4,967 kg)
- **Loaded weight**: 15,198 lb (6,894 kg)
- **Wing span**: 37.1 ft (11.3 m)
- **Wing area**: 287.9 ft² (26.8 m²)

### MiG-15bis
- **Engine**: Klimov VK-1 (5,952 lbf / 26.5 kN thrust)
- **Maximum speed**: 668 mph (1,075 km/h, Mach 0.87) at sea level
- **Service ceiling**: 50,850 ft (15,500 m)
- **Rate of climb**: 10,100 ft/min (51.3 m/s)
- **Wing loading**: 45.9 lb/ft² (224 kg/m²)
- **Thrust/weight**: 0.54
- **Armament**: 1× 37mm N-37 cannon (40 rounds), 2× 23mm NR-23 cannons (160 rounds)
- **Empty weight**: 7,900 lb (3,584 kg)
- **Loaded weight**: 11,177 lb (5,070 kg)
- **Wing span**: 33.1 ft (10.1 m)
- **Wing area**: 221.7 ft² (20.6 m²)

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Usage

```python
from simulator import F86Sabre, MiG15, EngagementSimulation

# Create aircraft
sabre = F86Sabre(position=[0, 0, -10000], velocity=[200, 0, 0])
mig = MiG15(position=[5000, 1000, -10000], velocity=[180, 0, 0])

# Run engagement
sim = EngagementSimulation(sabre, mig)
result = sim.run(max_time=300.0)

# Analyze results
result.plot_trajectories()
result.get_statistics()
```

### Monte Carlo Simulations

```python
from simulator import MonteCarloRunner

# Run 1000 engagements with varying initial conditions
runner = MonteCarloRunner(
    aircraft_types=['F86', 'MiG15'],
    num_simulations=1000,
    altitude_range=(10000, 40000),
    speed_range=(0.6, 0.85)  # Mach number
)

results = runner.run_parallel(n_cores=8)
results.plot_statistics()
results.export_csv('engagement_results.csv')
```

## Project Structure

```
simulator/
├── physics/              # Core physics engine
│   ├── vector.py        # 3D vector operations
│   ├── quaternion.py    # Rotation handling
│   └── atmosphere.py    # ISA atmosphere model
├── aerodynamics/        # Aerodynamic models
│   ├── forces.py        # Lift, drag, thrust
│   ├── coefficients.py  # Aerodynamic coefficients
│   └── tables.py        # Empirical data tables
├── aircraft/            # Aircraft implementations
│   ├── base.py          # Base aircraft class
│   ├── f86.py           # F-86 Sabre
│   └── mig15.py         # MiG-15
├── weapons/             # Weapon systems
│   ├── guns.py          # Gun ballistics
│   └── damage.py        # Damage modeling
├── simulation/          # Simulation engine
│   ├── integrator.py    # RK4 integration
│   ├── engagement.py    # Combat scenarios
│   └── monte_carlo.py   # MC framework
└── visualization/       # Plotting and analysis
    ├── plots.py         # 2D/3D visualization
    └── analysis.py      # Statistical analysis
```

## Physics Model

### Coordinate Systems
- **World frame**: North-East-Down (NED)
- **Body frame**: X-forward, Y-right, Z-down
- **Wind frame**: Aligned with velocity vector

### Forces
- **Aerodynamic**: Lift, drag (profile + induced)
- **Thrust**: Engine performance curves
- **Weight**: Gravitational force
- **Gun recoil**: Momentum transfer from firing

### Integration
- **Method**: 4th-order Runge-Kutta (RK4)
- **Time step**: Adaptive, typically 0.01-0.05 seconds
- **State vector**: Position, velocity, orientation (quaternion), angular velocity

## Validation

The simulator is validated against:
- Documented performance specifications
- Korean War engagement data
- NACA/NASA wind tunnel data
- Flight manual performance charts

## Technical References

1. **F-86 Technical Manual**: T.O. 1F-86A-1, USAF
2. **MiG-15 Technical Data**: Russian Aviation Museum archives
3. **"No Guts, No Glory"**: USAF Korean War air combat analysis
4. **NACA Reports**: Transonic aerodynamics research (1940s-1950s)
5. **"Sabre vs MiG-15"**: Osprey Duel Series analysis
6. **Stevens & Lewis**: "Aircraft Control and Simulation" (3rd ed.)

## License

MIT
