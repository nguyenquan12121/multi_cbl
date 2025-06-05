import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from kwave.kmedium import kWaveMedium
from kwave.kWaveSimulation import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.kgrid import kWaveGrid
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.dotdictionary import dotdict
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.utils.signals import tone_burst
from scipy.ndimage import binary_erosion
from skimage.morphology import disk
from kwave.utils.data import scale_time


def inpolygon(x, y, poly_x, poly_y):
    # Create Path object from polygon vertices
    polygon = np.vstack((poly_x, poly_y)).T
    path = Path(polygon)
    
    # Convert points to correct shape
    points = np.vstack((x.ravel(), y.ravel())).T
    
    # Test points
    inside = path.contains_points(points)
    
    # Reshape to match input
    return inside.reshape(x.shape)
def scaleSI(x):
    prefixes = [
        (1e-24, 'y', 'yocto'),
        (1e-21, 'z', 'zepto'),
        (1e-18, 'a', 'atto'),
        (1e-15, 'f', 'femto'),
        (1e-12, 'p', 'pico'),
        (1e-9, 'n', 'nano'),
        (1e-6, 'μ', 'micro'),
        (1e-3, 'm', 'milli'),
        (1e0, '', ''),
        (1e3, 'k', 'kilo'),
        (1e6, 'M', 'mega'),
        (1e9, 'G', 'giga'),
        (1e12, 'T', 'tera'),
        (1e15, 'P', 'peta'),
        (1e18, 'E', 'exa'),
        (1e21, 'Z', 'zetta'),
        (1e24, 'Y', 'yotta')
    ]
    
    # Handle arrays by taking max absolute value
    if isinstance(x, (np.ndarray, list)):
        x = np.max(np.abs(x))
    
    x = float(x)
    abs_x = abs(x)
    
    # Find the appropriate scale
    scale = 1.0
    prefix = ''
    prefix_fullname = ''
    
    if abs_x == 0:
        # Special case for zero
        return ('0', 1.0, '', '')
    
    for exp, p, pname in reversed(prefixes):
        if abs_x >= exp or np.isclose(abs_x, exp):
            scale = 1.0 / exp
            prefix = p
            prefix_fullname = pname
            break
    
    # Format the scaled value
    scaled_value = x * scale
    
    # Format string (2 decimal places for values < 100, 1 for larger)
    if 0 < abs(scaled_value) < 100:
        x_sc = f"{scaled_value:.2f}{prefix}"
    else:
        x_sc = f"{scaled_value:.1f}{prefix}"
    
    # Clean up formatting (remove .0 when not needed)
    if x_sc.endswith('.0'):
        x_sc = x_sc[:-2]
    if x_sc.endswith('.00'):
        x_sc = x_sc[:-3]
    
    return (x_sc, scale, prefix) 
def main():

    karray = kWaveArray()
    # Parameters
    PML_size = 20
    Nx = 128 - 2 * PML_size
    Ny = 128 - 2 * PML_size
    dx = 3 / Nx
    dy = 3 / Ny

    # Create grid
    grid = kWaveGrid([Nx, Ny], [dx, dy])

    # Medium properties
    c_air = 330
    rho_air = 1.225
    c_object = 1000
    rho_object = 1000
    wall_thickness = 2

    # Create hollow hexagon object
    object_mask = np.zeros((Nx, Ny))

    # Define hexagon vertices
    center_x = Nx / 2
    center_y = Ny / 2
    radius = round(min(Nx, Ny) * 0.3)

    theta = np.linspace(0, 2 * np.pi, 7)  # 6 vertices + 1 to close polygon
    hex_x = center_x + radius * np.cos(theta)
    hex_y = center_y + radius * np.sin(theta)

    # Create meshgrid and check points inside hexagon
    x, y = np.meshgrid(np.arange(Ny), np.arange(Nx))
    # points = np.vstack((x.ravel(), y.ravel())).T
    # hex_path = Path(np.vstack((hex_y, hex_x)).T)  # Note: x and y swapped for matplotlib
    # in_hex = hex_path.contains_points(points).reshape(Nx, Ny)
    in_hex = inpolygon(x, y, hex_y, hex_x);    
    object_mask[in_hex] = 1

    # Hollow out the hexagon
    se = disk(wall_thickness)
    object_mask = object_mask - binary_erosion(object_mask, structure=se)
    object_mask[object_mask < 0] = 0

    # Assign medium properties
    medium = kWaveMedium(
        sound_speed=c_air * np.ones((Nx, Ny))
    )
    medium.sound_speed[object_mask == 1] = c_object
    medium.density[object_mask == 1] = rho_object
    medium.sound_speed_ref = c_air

    # Time array
    cfl = 0.2
    t_end = 20e-3
    kgrid_t_array, dt = grid.makeTime(medium.sound_speed, cfl , t_end)

    # Source in the middle of the grid
    source_mag = 100
    sampling_freq = 1 / dt
    tone_burst_freq = 1.5e3
    tone_burst_cycles = 3

    source = kSource()
    source.p = source_mag * tone_burst(sampling_freq, tone_burst_freq, tone_burst_cycles)

    source.p_mask = np.zeros((Nx, Ny))
    source.p_mask[round(Nx/2), round(Ny/2)] = 1

    # Sensor records everywhere
    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Ny))
    sensor.mask[round(Nx/2), round(Ny/2)] = 1  # same as source

    display_mask = object_mask.copy()

    simulation_options = SimulationOptions(
        pml_inside=False,
        pml_size=PML_size,
        data_cast='single',
        save_to_disk=True,
        data_recast=True 
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
    sensor_data = kspace_first_order_2d_gpu(
        kgrid=grid,
        source=source,
        sensor=sensor,
        medium=medium,
        simulation_options=simulation_options,
        execution_options=execution_options
    )
    # # Now reshape will work properly
    # p_field = np.reshape(sensor_data['p'], (grid.Nt, Nx, Ny))
    # p_field = np.transpose(p_field, (0, 2, 1))  # Adjust orientation
    
    # # Normalize frames
    # max_value = np.max(np.abs(p_field))
    # normalized_frames = p_field / max_value
    
    # # Create animation
    # fig, ax = plt.subplots(figsize=(10, 8))
    
    # # Add hexagon outline
    # ax.plot(hex_x, hex_y, 'k-', linewidth=1)
    
    # # Initial frame
    # img = ax.imshow(normalized_frames[0], 
    #                cmap=plt.cm.seismic,
    #                norm=colors.Normalize(vmin=-1, vmax=1),
    #                extent=[0, Ny*dy*1e3, Nx*dx*1e3, 0])
    
    # # Colorbar
    # cbar = fig.colorbar(img, ax=ax)
    # cbar.set_label('Normalized Pressure')
    
    # # Animation function
    # def update(frame):
    #     img.set_array(normalized_frames[frame])
    #     ax.set_title(f'Frame {frame+1}/{grid.Nt}')
    #     return [img]
    
    # # Create and save animation
    # ani = FuncAnimation(fig, update, frames=min(grid.Nt, 100), interval=50, blit=True)
    # ani.save("wave_propagation.mp4", writer='ffmpeg', fps=15, dpi=150)
    # After simulation, create the animation
    # First, we need to get the pressure field data
    # Note: In k-Wave Python, the movie data is stored in sensor_data['p_frames']
    # If not available, we might need to modify the simulation options
    
   # Plot the recorded data
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # Get time scaling
    t_sc, scale, prefix = scaleSI(np.max(kgrid_t_array))
    # print(source.p)
    # Plot source signal
    ax1.plot(source.p, 'b-', linewidth=2)
    ax1.set_xlabel(f'Time [{prefix}s]')
    ax1.set_ylabel('Signal Amplitude')
    ax1.set_title('Source Pressure Signal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, len(source.p)])

    # Plot sensor signal
    ax2.plot((kgrid_t_array * scale).flatten(), sensor_data['p'], 'r-', linewidth=2)
    ax2.set_xlabel(f'Time [{prefix}s]')
    ax2.set_ylabel('Signal Amplitude')
    ax2.set_title('Sensor Pressure Signal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, np.max(kgrid_t_array * scale)])

    plt.tight_layout()
    plt.show()
    
    # Optional: Visualize the hexagon geometry
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(object_mask.T, extent=[0, Nx*dx, 0, Ny*dy], origin='lower', cmap='gray')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title('Hollow Hexagon Object')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Object Mask')
    plt.show()

if __name__ == "__main__":
    main()