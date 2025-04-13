import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from tabulate import tabulate

# Physical parameters
g = 9.81  # Gravity (m/s²)
rho = 1.225  # Air density (kg/m³)
sheet_width = 0.20  # Width of sheet (m)
sheet_height = 0.15  # Height of sheet (m)
sheet_mass = 0.010  # Mass of sheet (kg) - assuming light plastic
sheet_thickness = 0.001  # Thickness (m)
drop_height = 1.5  # Total height to fall (m)

# Simulation parameters
dt = 0.01  # Time step (s)(defines the scale of the graphs and animation while getting values for the simulation)
initial_angle = np.pi / 6  # Initial angle (30 degrees)(basically the angle of the sheet when it is dropped)
initial_angular_velocity = 0  # Initial angular velocity (rad/s)
max_time = 2.0  # Maximum simulation time (s)

# Turbulence parameters
turbulence_intensity = 0.15  # Turbulence intensity (0-1)
turbulence_length_scale = 0.1  # Characteristic length scale of turbulence (m)
turbulence_time_scale = 0.05  # Time scale for turbulence fluctuations (s)

# Calculate moment of inertia for a rectangular sheet
moment_of_inertia = (sheet_mass * (sheet_width**2 + sheet_height**2)) / 12

# Function to calculate drag coefficient based on angle of attack
def get_drag_coefficient(angle, reynolds_number):
    # Enhanced model: drag coefficient varies with angle and Reynolds number
    # Maximum when perpendicular to flow, minimum when parallel
    normalized_angle = np.abs(np.sin(angle))
    
    # Reynolds number effect (simplified model)
    re_factor = 1.0
    if reynolds_number < 1000:
        re_factor = 1.2  # Higher drag at low Reynolds numbers
    elif reynolds_number > 100000:
        re_factor = 0.9  # Lower drag at high Reynolds numbers
        
    return (1.28 * normalized_angle + 0.1) * re_factor

# Function to calculate lift coefficient based on angle of attack
def get_lift_coefficient(angle, reynolds_number):
    
    
    # Normalize angle to [-π/2, π/2]
    effective_angle = angle
    while effective_angle > np.pi/2:
        effective_angle -= np.pi
    while effective_angle < -np.pi/2:
        effective_angle += np.pi
    
    # Thin airfoil approximation with stall
    if abs(effective_angle) < np.pi/6:  # Linear region (< 30 degrees)
        cl = 2.0 * np.pi * np.sin(effective_angle)
    else:  # Stall region
        # Gradual decrease after stall
        sign = 1 if effective_angle > 0 else -1
        cl = sign * (1.0 + np.cos(2 * effective_angle))
    
    # Reynolds number effect (simplified)
    if reynolds_number < 1000:
        cl *= 0.8  # Reduced lift at low Reynolds numbers
    
    return cl

# Function to calculate projected area based on orientation
def get_projected_area(angle):
    # Area projected in the direction of motion
    return (sheet_width * sheet_height * np.abs(np.sin(angle)) + 
            sheet_width * sheet_thickness * np.abs(np.cos(angle)))

# Function to generate turbulent velocity fluctuations
def generate_turbulence(time, prev_fluctuation=None):
    if prev_fluctuation is None:
        # Initialize with random fluctuations
        u_fluct = np.random.normal(0, turbulence_intensity, 2)
    else:
        # Time-correlated turbulence using a simple AR(1) process
        correlation = np.exp(-dt / turbulence_time_scale)
        random_component = np.random.normal(0, turbulence_intensity * np.sqrt(1 - correlation**2), 2)
        u_fluct = correlation * prev_fluctuation + random_component
    
    return u_fluct

# Main simulation function
def simulate_falling_sheet():
    # Initial conditions
    x = 0  # Horizontal position (m)
    y = drop_height  # Vertical position (m)
    vx = 0  # Horizontal velocity (m/s)
    vy = 0  # Vertical velocity (m/s)
    angle = initial_angle  # Orientation angle (rad)
    angular_velocity = initial_angular_velocity  # Angular velocity (rad/s)
    turbulence_fluct = None  # Initial turbulence fluctuation
    
    # Arrays to store data for plotting
    times = []
    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []
    velocity_magnitudes = []
    angles = []
    angular_velocities = []
    angular_momenta = []
    projected_areas = []
    drag_forces = []
    lift_forces = []
    reynolds_numbers = []
    turbulence_x = []
    turbulence_y = []
    
    t = 0
    step = 0
    
    # Run simulation until sheet reaches ground or max time
    while y > 0 and t < max_time:
        # Store current state
        times.append(t)
        positions_x.append(x)
        positions_y.append(y)
        velocities_x.append(vx)
        velocities_y.append(vy)
        velocity_magnitudes.append(np.sqrt(vx**2 + vy**2))
        angles.append(angle)
        angular_velocities.append(angular_velocity)
        angular_momenta.append(moment_of_inertia * angular_velocity)
        
        # Calculate current projected area
        area = get_projected_area(angle)
        projected_areas.append(area)
        
        # Calculate velocity magnitude
        v = np.sqrt(vx**2 + vy**2)
        
        # Calculate Reynolds number
        characteristic_length = max(sheet_width, sheet_height)
        kinematic_viscosity = 1.5e-5  # Air at room temperature (m²/s)
        reynolds = (v * characteristic_length) / kinematic_viscosity if v > 0.001 else 0
        reynolds_numbers.append(reynolds)
        
        # Generate turbulence fluctuations
        turbulence_fluct = generate_turbulence(t, turbulence_fluct)
        turbulence_x.append(turbulence_fluct[0])
        turbulence_y.append(turbulence_fluct[1])
        
        # Apply turbulence to velocity
        effective_vx = vx + turbulence_fluct[0] * v
        effective_vy = vy + turbulence_fluct[1] * v
        effective_v = np.sqrt(effective_vx**2 + effective_vy**2)
        
        # Calculate effective angle of attack
        effective_angle = np.arctan2(effective_vy, effective_vx) - angle if effective_v > 0.001 else 0
        
        # Calculate drag and lift coefficients
        cd = get_drag_coefficient(effective_angle, reynolds)
        cl = get_lift_coefficient(effective_angle, reynolds)
        
        # Calculate drag and lift forces
        dynamic_pressure = 0.5 * rho * effective_v**2
        drag_magnitude = dynamic_pressure * cd * area
        lift_magnitude = dynamic_pressure * cl * area
        
        # Calculate force components
        # Drag is opposite to velocity direction
        if effective_v > 0.001:
            drag_fx = -drag_magnitude * effective_vx / effective_v
            drag_fy = -drag_magnitude * effective_vy / effective_v
            
            # Lift is perpendicular to velocity direction
            # Calculate unit vector perpendicular to velocity
            perp_x = -effective_vy / effective_v
            perp_y = effective_vx / effective_v
            
            # Apply lift force in the perpendicular direction
            lift_fx = lift_magnitude * perp_x
            lift_fy = lift_magnitude * perp_y
        else:
            drag_fx = drag_fy = lift_fx = lift_fy = 0
        
        # Total forces
        fx = drag_fx + lift_fx
        fy = drag_fy + lift_fy - sheet_mass * g  # Include gravity
        
        drag_forces.append(np.sqrt(drag_fx**2 + drag_fy**2))
        lift_forces.append(np.sqrt(lift_fx**2 + lift_fy**2))
        
        
        cp_offset = 0.1 * sheet_width * np.sin(2 * effective_angle)  # Simplified CP model
        
        # Calculate moment arm from center of mass to center of pressure
        torque_arm = cp_offset
        
        # Calculate torque from lift and drag
        torque_lift = lift_magnitude * torque_arm
        
        # Add turbulence-induced torque (random fluctuations)
        turbulence_torque = 0.001 * sheet_mass * turbulence_intensity * np.random.normal(0, 1)
        
        # Total torque
        torque = torque_lift + turbulence_torque
        
        # Update velocities (Euler integration)
        vx += (fx / sheet_mass) * dt
        vy += (fy / sheet_mass) * dt
        angular_velocity += (torque / moment_of_inertia) * dt
        
        # Update positions and angle
        x += vx * dt
        y += vy * dt
        angle += angular_velocity * dt
        
        # Normalize angle to [-π, π]
        angle = ((angle + np.pi) % (2 * np.pi)) - np.pi
        
        t += dt
        step += 1
    
    return {
        'times': np.array(times),
        'positions_x': np.array(positions_x),
        'positions_y': np.array(positions_y),
        'velocities_x': np.array(velocities_x),
        'velocities_y': np.array(velocities_y),
        'velocity_magnitudes': np.array(velocity_magnitudes),
        'angles': np.array(angles),
        'angular_velocities': np.array(angular_velocities),
        'angular_momenta': np.array(angular_momenta),
        'projected_areas': np.array(projected_areas),
        'drag_forces': np.array(drag_forces),
        'lift_forces': np.array(lift_forces),
        'reynolds_numbers': np.array(reynolds_numbers),
        'turbulence_x': np.array(turbulence_x),
        'turbulence_y': np.array(turbulence_y)
    }

# Run simulation
print("Running simulation with enhanced lift, rotation, and turbulence models...")
results = simulate_falling_sheet()
print(f"Simulation completed with {len(results['times'])} time steps")

# Create plots with improved layout
def create_plots(results):
    # Set up figure with subplots
    plt.figure(figsize=(15, 15))
    
    # Use tight_layout and adjust spacing
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot 1: Trajectory
    plt.subplot(3, 3, 1)
    plt.plot(results['positions_x'], results['positions_y'], 'b-', linewidth=2)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Sheet Trajectory')
    plt.grid(True)
    
    # Plot 2: Position vs Time
    plt.subplot(3, 3, 2)
    plt.plot(results['times'], results['positions_y'], 'r-', label='Y Position')
    plt.plot(results['times'], results['positions_x'], 'g-', label='X Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Velocity vs Time
    plt.subplot(3, 3, 3)
    plt.plot(results['times'], results['velocity_magnitudes'], 'k-', label='Magnitude')
    plt.plot(results['times'], results['velocities_x'], 'g-', label='X Component')
    plt.plot(results['times'], results['velocities_y'], 'r-', label='Y Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Orientation Angle vs Time
    plt.subplot(3, 3, 4)
    plt.plot(results['times'], np.degrees(results['angles']), 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Orientation Angle vs Time')
    plt.grid(True)
    
    # Plot 5: Angular Velocity vs Time
    plt.subplot(3, 3, 5)
    plt.plot(results['times'], results['angular_velocities'], 'c-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity vs Time')
    plt.grid(True)
    
    # Plot 6: Forces vs Time
    plt.subplot(3, 3, 6)
    plt.plot(results['times'], results['drag_forces'], 'r-', label='Drag Force')
    plt.plot(results['times'], results['lift_forces'], 'b-', label='Lift Force')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.title('Aerodynamic Forces vs Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 7: Reynolds Number vs Time
    plt.subplot(3, 3, 7)
    plt.plot(results['times'], results['reynolds_numbers'], 'm-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Reynolds Number')
    plt.title('Reynolds Number vs Time')
    plt.grid(True)
    
    # Plot 8: Turbulence Fluctuations
    plt.subplot(3, 3, 8)
    plt.plot(results['times'], results['turbulence_x'], 'g-', label='X Component')
    plt.plot(results['times'], results['turbulence_y'], 'r-', label='Y Component')
    plt.xlabel('Time (s)')
    plt.ylabel('Turbulence Intensity')
    plt.title('Turbulence Fluctuations')
    plt.legend()
    plt.grid(True)
    
    # Plot 9: Lift to Drag Ratio
    plt.subplot(3, 3, 9)
    with np.errstate(divide='ignore', invalid='ignore'):
        lift_drag_ratio = np.divide(results['lift_forces'], results['drag_forces'])
        lift_drag_ratio[~np.isfinite(lift_drag_ratio)] = 0  # Replace inf/NaN with 0
    plt.plot(results['times'], lift_drag_ratio, 'k-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('L/D Ratio')
    plt.title('Lift to Drag Ratio')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('falling_sheet_plots.png', dpi=300)
    print("Plots saved as 'falling_sheet_plots.png'")
    plt.show()
    
    # Create 3D visualization of the sheet's motion with rotation
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization (not all points to avoid clutter)
    sample_indices = np.linspace(0, len(results['times'])-1, 15, dtype=int)
    
    for i in sample_indices:
        # Create a rectangle representing the sheet
        angle = results['angles'][i]
        x = results['positions_x'][i]
        y = results['positions_y'][i]
        
        # Calculate corners of the rectangle
        corners = []
        for dx, dy in [(-sheet_width/2, -sheet_height/2), 
                       (sheet_width/2, -sheet_height/2),
                       (sheet_width/2, sheet_height/2), 
                       (-sheet_width/2, sheet_height/2)]:
            # Rotate the corner
            rotated_dx = dx * np.cos(angle) - dy * np.sin(angle)
            rotated_dy = dx * np.sin(angle) + dy * np.cos(angle)
            corners.append((x + rotated_dx, y + rotated_dy, 0))
        
        # Plot the rectangle
        for j in range(4):
            ax.plot([corners[j][0], corners[(j+1)%4][0]],
                    [corners[j][1], corners[(j+1)%4][1]],
                    [corners[j][2], corners[(j+1)%4][2]], 'r-')
    
    # Plot trajectory
    ax.plot(results['positions_x'], results['positions_y'], np.zeros_like(results['times']), 'b-', alpha=0.5)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('3D Visualization of Sheet Motion with Rotation')
    
    plt.savefig('falling_sheet_3d.png', dpi=300)
    print("3D visualization saved as 'falling_sheet_3d.png'")
    plt.show()

# Create animation of falling sheet with rotation and forces
def create_animation(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(min(results['positions_x'])-0.5, max(results['positions_x'])+0.5)
    ax.set_ylim(0, drop_height+0.5)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Falling Sheet Animation with Forces')
    ax.grid(True)
    
    # Create rectangle patch
    rect = Rectangle((0, 0), sheet_width, sheet_height, 
                     angle=np.degrees(results['angles'][0]), 
                     color='blue', alpha=0.7)
    rect.set_xy((-sheet_width/2, -sheet_height/2))  # Center the rectangle
    
    # Add rectangle to plot
    patch = ax.add_patch(rect)
    
    # Create trajectory line
    line, = ax.plot([], [], 'r-', alpha=0.5)
    
    # Create force arrows (drag and lift)
    drag_arrow = ax.arrow(0, 0, 0, 0, head_width=0.02, head_length=0.03, fc='red', ec='red', alpha=0.7)
    lift_arrow = ax.arrow(0, 0, 0, 0, head_width=0.02, head_length=0.03, fc='green', ec='green', alpha=0.7)
    
    # Text for time and force display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    force_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    
    def init():
        patch.set_xy((-sheet_width/2, -sheet_height/2))
        line.set_data([], [])
        time_text.set_text('')
        force_text.set_text('')
        drag_arrow.set_visible(False)
        lift_arrow.set_visible(False)
        return patch, line, time_text, force_text, drag_arrow, lift_arrow
    
    def update(frame):
        # Update rectangle position and orientation
        angle = results['angles'][frame]
        x = results['positions_x'][frame]
        y = results['positions_y'][frame]
        
        # Set rectangle position and rotation
        transform = plt.matplotlib.transforms.Affine2D() \
            .rotate(angle) \
            .translate(x, y) + ax.transData
        patch.set_transform(transform)
        
        # Update trajectory line
        line.set_data(results['positions_x'][:frame+1], results['positions_y'][:frame+1])
        
        # Update time and force text
        time_text.set_text(f'Time: {results["times"][frame]:.2f} s')
        force_text.set_text(f'Drag: {results["drag_forces"][frame]:.3f} N, Lift: {results["lift_forces"][frame]:.3f} N')
        
        # Update force arrows
        # Calculate velocity direction for drag
        vx = results['velocities_x'][frame]
        vy = results['velocities_y'][frame]
        v_mag = np.sqrt(vx**2 + vy**2)
        
        if v_mag > 0.001:
            # Drag force (opposite to velocity)
            drag_scale = 0.2  # Scale factor for visualization
            drag_mag = results['drag_forces'][frame]
            drag_x = -drag_scale * drag_mag * vx / v_mag
            drag_y = -drag_scale * drag_mag * vy / v_mag
            
            # Lift force (perpendicular to velocity)
            lift_scale = 0.2  # Scale factor for visualization
            lift_mag = results['lift_forces'][frame]
            lift_x = lift_scale * lift_mag * (-vy) / v_mag
            lift_y = lift_scale * lift_mag * vx / v_mag
            
            # Update arrows
            drag_arrow.set_data(x=x, y=y, dx=drag_x, dy=drag_y)
            lift_arrow.set_data(x=x, y=y, dx=lift_x, dy=lift_y)
            drag_arrow.set_visible(True)
            lift_arrow.set_visible(True)
        else:
            drag_arrow.set_visible(False)
            lift_arrow.set_visible(False)
        
        return patch, line, time_text, force_text, drag_arrow, lift_arrow
    
    # Create animation
    frames = min(100, len(results['times']))  # Limit frames for performance
    indices = np.linspace(0, len(results['times'])-1, frames, dtype=int)
    
    anim = FuncAnimation(fig, update, frames=indices, init_func=init, 
                         blit=True, interval=50)
    
    # Save animation
    try:
        anim.save('falling_sheet_animation.gif', writer='pillow', fps=20, dpi=100)
        print("Animation saved as 'falling_sheet_animation.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    plt.show()

# Generate a comprehensive report
def generate_report():
    report = """
# Enhanced Falling Rectangular Sheet Analysis Report

## Problem Statement
A rectangular plastic sheet of size 20 cm x 15 cm is falling freely in air. The sheet is rigid and not flexible, 
but free to rotate or reorient in any direction. This report analyzes the sheet's position, orientation, and other 
physical quantities as it falls down a height of 1.5 m, with special focus on lift forces, rotation dynamics, and 
turbulence effects.

## Physical Model

### Sheet Properties:
- Dimensions: 20 cm × 15 cm × 0.1 cm
- Mass: 10 grams
- Moment of Inertia: {:.6f} kg·m²

### Initial Conditions:
- Initial Position: (0, 1.5) m
- Initial Velocity: (0, 0) m/s
- Initial Orientation: {:.1f} degrees
- Initial Angular Velocity: 0 rad/s

### Forces and Effects Considered:
1. **Gravity**: Constant downward force (m·g)
2. **Drag Force**: Depends on projected area, drag coefficient, velocity squared, and Reynolds number
3. **Lift Force**: Depends on angle of attack, projected area, lift coefficient, and velocity squared
4. **Rotation**: Full rotational dynamics with torque from aerodynamic forces
5. **Turbulence**: Random fluctuations in air velocity with temporal correlation

## Enhanced Aerodynamic Model

### Lift Force Model:
- Lift coefficient based on thin airfoil theory with stall effects
- Reynolds number dependence incorporated
- Center of pressure shift with angle of attack

### Rotation Dynamics:
- Torque calculated from pressure distribution and center of pressure location
- Moment of inertia calculated for rectangular plate
- Full coupling between translational and rotational motion

### Turbulence Model:
- Intensity: {:.2f} (fraction of mean velocity)
- Length scale: {:.2f} m
- Time scale: {:.2f} s
- Time-correlated random fluctuations using AR(1) process
- Effects on both translational and rotational dynamics

## Results Summary

### Motion Analysis:
- Total Fall Time: {:.2f} seconds
- Maximum Horizontal Displacement: {:.2f} m
- Maximum Velocity: {:.2f} m/s
- Maximum Angular Velocity: {:.2f} rad/s
- Maximum Angular Momentum: {:.4f} kg·m²/s
- Maximum Reynolds Number: {:.0f}
- Maximum Lift Force: {:.4f} N
- Maximum Drag Force: {:.4f} N

### Key Observations:
1. The sheet initially accelerates due to gravity while experiencing minimal drag and lift.
2. As velocity increases, aerodynamic forces become more significant, affecting both linear and rotational motion.
3. Lift forces create substantial lateral movement and induce rotation.
4. Turbulence causes irregular fluctuations in the sheet's trajectory and orientation.
5. The sheet's orientation continuously changes due to torque from asymmetric aerodynamic forces.
6. The projected area varies with orientation, which in turn affects the drag and lift forces.
7. Reynolds number effects modify the aerodynamic coefficients throughout the fall.

## Conclusion
The motion of a falling rigid sheet is complex due to the coupling between translational and rotational dynamics,
further complicated by lift forces and turbulence effects. The sheet follows an irregular curved trajectory while
rotating in a non-uniform manner. The simulation demonstrates how aerodynamic forces significantly influence the
motion of lightweight objects falling through air, and how turbulence adds unpredictability to the motion.

## Appendix: Simulation Details
The simulation uses a time step of {:.3f} seconds and employs Euler integration to solve the coupled differential 
equations of motion. The aerodynamic coefficients are modeled as functions of the effective angle of attack and
Reynolds number. Turbulence is modeled as time-correlated random fluctuations with specified intensity and time scale.
""".format(
        moment_of_inertia,
        np.degrees(initial_angle),
        turbulence_intensity,
        turbulence_length_scale,
        turbulence_time_scale,
        results['times'][-1],
        max(np.abs(results['positions_x'])),
        max(results['velocity_magnitudes']),
        max(np.abs(results['angular_velocities'])),
        max(np.abs(results['angular_momenta'])),
        max(results['reynolds_numbers']),
        max(results['lift_forces']),
        max(results['drag_forces']),
        dt
    )
    
    with open('falling_sheet_enhanced_report.md', 'w') as f:
        f.write(report)
    
    print("\nEnhanced report generated and saved as 'falling_sheet_enhanced_report.md'")

# Create data table with lift, rotation, and turbulence analysis
def create_enhanced_data_table(results):
    # Sample data at regular intervals for display
    num_samples = 10
    indices = np.linspace(0, len(results['times'])-1, num_samples, dtype=int)
    
    # Prepare data for tabulate
    headers = ["Time (s)", "Y Pos (m)", "X Pos (m)", "Vel (m/s)", "Angle (°)", 
               "Ang Vel (rad/s)", "Lift (N)", "Drag (N)", "Reynolds", "Turb X", "Turb Y"]
    table_data = []
    
    for i in indices:
        table_data.append([
            f"{results['times'][i]:.3f}",
            f"{results['positions_y'][i]:.3f}",
            f"{results['positions_x'][i]:.3f}",
            f"{results['velocity_magnitudes'][i]:.3f}",
            f"{np.degrees(results['angles'][i]):.1f}",
            f"{results['angular_velocities'][i]:.3f}",
            f"{results['lift_forces'][i]:.5f}",
            f"{results['drag_forces'][i]:.5f}",
            f"{results['reynolds_numbers'][i]:.0f}",
            f"{results['turbulence_x'][i]:.3f}",
            f"{results['turbulence_y'][i]:.3f}"
        ])
    
    # Print table with tabulate for better formatting
    print("\nEnhanced Data Table for Falling Sheet Simulation (with Lift, Rotation, and Turbulence):")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save complete data to CSV file
    with open('falling_sheet_enhanced_data.csv', 'w') as f:
        f.write(",".join(headers) + "\n")
        for i in range(len(results['times'])):
            f.write(f"{results['times'][i]:.3f},{results['positions_y'][i]:.3f},{results['positions_x'][i]:.3f},"
                   f"{results['velocity_magnitudes'][i]:.3f},{np.degrees(results['angles'][i]):.1f},"
                   f"{results['angular_velocities'][i]:.3f},{results['lift_forces'][i]:.5f},"
                   f"{results['drag_forces'][i]:.5f},{results['reynolds_numbers'][i]:.0f},"
                   f"{results['turbulence_x'][i]:.3f},{results['turbulence_y'][i]:.3f}\n")
    
    print(f"\nComplete enhanced data saved to 'falling_sheet_enhanced_data.csv' ({len(results['times'])} rows)")
    
    # Also save a formatted version for easy viewing
    with open('falling_sheet_enhanced_data_formatted.txt', 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("Formatted enhanced table saved to 'falling_sheet_enhanced_data_formatted.txt'")

# Main execution
print("\n" + "="*80)
print("ENHANCED FALLING RECTANGULAR SHEET SIMULATION".center(80))
print("="*80 + "\n")

print("Physical Parameters:")
print(f"  - Sheet dimensions: {sheet_width*100:.1f} cm × {sheet_height*100:.1f} cm × {sheet_thickness*100:.1f} cm")
print(f"  - Sheet mass: {sheet_mass*1000:.1f} g")
print(f"  - Initial angle: {np.degrees(initial_angle):.1f}°")
print(f"  - Drop height: {drop_height:.1f} m")
print(f"  - Time step: {dt:.3f} s")
print(f"  - Turbulence intensity: {turbulence_intensity:.2f}")
print("\n" + "-"*80 + "\n")

# Generate all outputs
create_enhanced_data_table(results)
create_plots(results)
create_animation(results)
generate_report()

print("\n" + "="*80)
print("ENHANCED SIMULATION COMPLETE".center(80))
print("="*80)