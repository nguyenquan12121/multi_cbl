clear all;
close all;

% -------------------------------------------------------------------------
% Parameters
% -------------------------------------------------------------------------
PML_size = 20;                   % thickness of the absorbing layer on the edges
Nx = 128 - 2*PML_size;           % number of grid points (x)
Ny = 128 - 2*PML_size;           % number of grid points (y)
dx = 3 / Nx;                     % spatial resolution in x [m]
dy = 3 / Ny;                     % spatial resolution in y [m]
kgrid = makeGrid(Nx, dx, Ny, dy);

% -------------------------------------------------------------------------
% Medium properties
% -------------------------------------------------------------------------
c_air = 330;                     % sound speed in air [m/s]
rho_air = 1.225;                 % density of air [kg/m^3]
c_object = 1000;                 % sound speed in ring object [m/s]
rho_object = 1000;               % density of ring object [kg/m^3]
thickness = 2;                   
outer_radius = 30;               % outer radius of ring in grid points

% -------------------------------------------------------------------------
% Create ring-shaped object
% -------------------------------------------------------------------------
ring_center_x = Nx / 2;
ring_center_y = Ny / 2;
[x, y] = meshgrid(1:Ny, 1:Nx);
dist_from_center = sqrt((x - ring_center_y).^2 + (y - ring_center_x).^2);
inner_radius = outer_radius - thickness;

object = dist_from_center >= inner_radius & dist_from_center <= outer_radius;

% Assign medium properties
medium.sound_speed = c_air * ones(Nx, Ny);
medium.sound_speed(object) = c_object;
medium.density = rho_air * ones(Nx, Ny);
medium.density(object) = rho_object;
medium.sound_speed_ref = c_air;

% -------------------------------------------------------------------------
% Create time array
% -------------------------------------------------------------------------
cfl = 0.2;
t_end = 20e-3;  % simulation time [s]
[kgrid.t_array, dt] = makeTime(kgrid, medium.sound_speed, cfl, t_end);

% -------------------------------------------------------------------------
% Define source (tone burst at center)
% -------------------------------------------------------------------------
source_mag = 100;                      % pressure magnitude [Pa]
sampling_freq = 1/dt;
tone_burst_freq = 1.5e3;             % Hz
tone_burst_cycles = 3;
source.p = source_mag * toneBurst(sampling_freq, tone_burst_freq, tone_burst_cycles);

source.p_mask = zeros(Nx, Ny);
source.p_mask(round(Nx/2), round(Ny/2)) = 1;  % source at center

% -------------------------------------------------------------------------
% Define sensor
% -------------------------------------------------------------------------
sensor.mask = zeros(Nx, Ny);
sensor.mask(round(Nx/2), round(Ny/2)) = 1;  % same as source

% -------------------------------------------------------------------------
% Run the simulation
% -------------------------------------------------------------------------
display_mask = object;

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

