clear all;
close all;

% Parameters
PML_size = 20;
Nx = 128 - 2*PML_size;
Ny = 128 - 2*PML_size;
dx = 3 / Nx;
dy = 3 / Ny;
kgrid = makeGrid(Nx, dx, Ny, dy);

% Medium properties
c_air = 330;
rho_air = 1.225;
c_object = 1000;
rho_object = 1000;
wall_thickness = 2;

% Create hollow hexagon object centered and scaled
object = zeros(Nx, Ny);

% Define hexagon vertices
center_x = Nx/2;
center_y = Ny/2;
radius = round(min(Nx, Ny) * 0.3);

theta = linspace(0, 2*pi, 7); % 6 vertices + 1 to close polygon
hex_x = center_x + radius * cos(theta);
hex_y = center_y + radius * sin(theta);

[x, y] = meshgrid(1:Ny, 1:Nx);
in_hex = inpolygon(x, y, hex_y, hex_x); % Note order (x,y) swapped for meshgrid coords
object(in_hex) = 1;

% Hollow out the hexagon 
se = strel('disk', wall_thickness, 0);
object = object - imerode(object, se);
object(object < 0) = 0;

% Assign medium properties
medium.sound_speed = c_air * ones(Nx, Ny);
medium.sound_speed(object == 1) = c_object;
medium.density = rho_air * ones(Nx, Ny);
medium.density(object == 1) = rho_object;
medium.sound_speed_ref = c_air;

% Time array 
cfl = 0.2;
t_end = 20e-3;  
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed, cfl, t_end);

% Source in the middle of the grid
source_mag = 100;
sampling_freq = 1/dt;
tone_burst_freq = 1.5e3;
tone_burst_cycles = 3;
source.p = source_mag * toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles);

source.p_mask = zeros(Nx, Ny);
source.p_mask(round(Nx/2), round(Ny/2)) = 1;

% Sensor records everywhere
sensor.mask = zeros(Nx, Ny);
sensor.mask(round(Nx/2), round(Ny/2)) = 1;  % same as source


display_mask = object;

% Run simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, ...
    'PMLInside', false, ...
    'PlotPML', false, ...
    'PMLSize', PML_size, ...
    'DisplayMask', display_mask, ...
    'PlotScale', [-1, 1]*source_mag, ...
    'DataCast', 'single');

% plot the recorded data
figure;
subplot(2, 1, 1);
[t_sc scale prefix] = scaleSI(max(kgrid.t_array(:)));
plot(source.p, 'b-');
xlabel(['Time [' prefix 's]']);
ylabel('Signal Amplitude');
axis tight;
title('Source Pressure Signal');

subplot(2, 1, 2), plot(kgrid.t_array*scale, sensor_data, 'r-');
xlabel(['Time [' prefix 's]']);
ylabel('Signal Amplitude');
axis tight;
title('Sensor Pressure Signal');
