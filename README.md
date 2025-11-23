# Air Combat Simulator - F-86 Sabre vs MiG-15

A highly accurate, physically realistic air combat simulator focusing on Korean War-era jet aircraft.

## Features

- **Physically accurate flight dynamics**: 6-DOF (Degrees of Freedom) equations of motion
- **Realistic aerodynamics**: Lift, drag, thrust modeling based on empirical data
- **Accurate aircraft models**: F-86 Sabre and MiG-15 with historical specifications
- **Weapon systems**: Period-accurate gun systems (M3 Browning .50 cal, NR-23 23mm)
- **Damage modeling**: Component-based damage system
- **Historical basis**: Uses documented performance data from Korean War engagements

## Aircraft Specifications

### F-86 Sabre
- Top speed: 687 mph (Mach 0.9) at sea level
- Service ceiling: 49,000 ft
- Rate of climb: 9,300 ft/min
- Armament: 6× .50 caliber M3 Browning machine guns
- Engine: General Electric J47-GE-27 (5,910 lbf thrust)

### MiG-15
- Top speed: 668 mph (Mach 0.87) at sea level
- Service ceiling: 50,850 ft
- Rate of climb: 10,100 ft/min
- Armament: 1× 37mm N-37, 2× 23mm NR-23 cannons
- Engine: Klimov VK-1 (5,952 lbf thrust)

## Installation

```bash
npm install
npm run build
```

## Usage

```bash
npm start
```

## Project Structure

```
src/
├── physics/          # Core physics engine (vectors, quaternions)
├── aerodynamics/     # Aerodynamic models and forces
├── aircraft/         # Aircraft specifications and models
├── weapons/          # Weapon systems and ballistics
├── simulation/       # Simulation engine and integration
└── index.ts          # Main entry point
```

## Technical Approach

The simulator uses:
- **SI units** throughout (meters, kilograms, seconds)
- **Right-handed coordinate system** (X: forward, Y: right, Z: down)
- **Quaternions** for rotation to avoid gimbal lock
- **RK4 integration** for numerical stability
- **Empirical aerodynamic data** from declassified sources

## References

- "The F-86 Sabre" - Technical Manual T.O. 1F-86A-1
- "MiG-15" - Russian Aviation Museum Technical Data
- "No Guts, No Glory" - USAF Korean War Air Combat Analysis
- NACA Reports on transonic aerodynamics
