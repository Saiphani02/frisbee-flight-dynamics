%% =========================================================
%  Flight Dynamics of a Frisbee (Disc Golf Disc)
%  Student : Sai Phanidhar Ammiraju
%  Course  : Flight Dynamics | Florida Atlantic University
%  Professor: Tom Lawrence
%
%  HOW TO RUN:
%    1. Open MATLAB
%    2. Open this file
%    3. Press Run (F5) or click the green Run button
%    4. All plots will appear automatically
% =========================================================
clear; clc; close all;

%% ── Physical & Aerodynamic Parameters ──────────────────
m    = 0.175;          % disc mass [kg]
R    = 0.107;          % disc radius [m]
A    = pi * R^2;       % planform area [m^2]
rho  = 1.225;          % air density [kg/m^3]
g    = 9.81;           % gravity [m/s^2]

% Aerodynamic coefficients (disc golf disc, empirical)
CL0  =  0.10;   CLa =  1.80;    % lift
CD0  =  0.08;   CDa =  2.72;    % drag
Cm0  =  0.01;   Cma = -0.40;    % pitching moment
Cmq  = -0.005;                  % pitch damping

% Inertia moments
Izz  = 0.5  * m * R^2;   % axial (spin axis)
Ixx  = 0.25 * m * R^2;   % transverse

% Pack all parameters into a struct for ODE function
params.m=m; params.R=R; params.A=A; params.rho=rho; params.g=g;
params.CL0=CL0; params.CLa=CLa; params.CD0=CD0; params.CDa=CDa;
params.Cm0=Cm0; params.Cma=Cma; params.Cmq=Cmq;
params.Izz=Izz; params.Ixx=Ixx;

%% ── Helper: Run one simulation ──────────────────────────
% Returns [t, x, y, z] arrays
function [t_out, x_out, y_out, z_out] = simulate(v0, angle_deg, spin_rpm, side_deg, params)
    if nargin < 4, side_deg = 0; end
    angle = deg2rad(angle_deg);
    side  = deg2rad(side_deg);
    spin  = spin_rpm * 2*pi/60;      % RPM → rad/s

    vx0 = v0 * cos(angle) * cos(side);
    vy0 = v0 * cos(angle) * sin(side);
    vz0 = v0 * sin(angle);

    % State: [x, y, z, vx, vy, vz, phi, theta, dphi, dtheta]
    state0 = [0; 0; 1.5; vx0; vy0; vz0; 0; angle; 0; 0];

    % Event: stop when z = 0 (disc hits ground)
    opts = odeset('Events', @ground_event, 'MaxStep', 0.02, ...
                  'RelTol', 1e-6, 'AbsTol', 1e-8);

    [t_out, Y] = ode45(@(t,s) frisbee_eom(t, s, spin, params), ...
                       [0 20], state0, opts);

    x_out = Y(:,1);
    y_out = Y(:,2);
    z_out = max(Y(:,3), 0);
end

%% ════════════════════════════════════════════════════════
%  PLOT 1 — Effect of Launch Angle
%% ════════════════════════════════════════════════════════
fprintf('Running Plot 1: Launch Angle Study...\n');
figure('Name','Effect of Launch Angle','NumberTitle','off','Color','w',...
       'Position',[100 100 1200 480]);

subplot(1,2,1);
angles_list = [5 10 15 20 25 30];
cmap1 = parula(length(angles_list));
hold on;
for k = 1:length(angles_list)
    [t,x,y,z] = simulate(20, angles_list(k), 600, 0, params);
    plot(x, z, 'Color', cmap1(k,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('%d deg', angles_list(k)));
end
hold off;
xlabel('Horizontal Distance (m)'); ylabel('Height (m)');
title('Trajectory Profiles (v_0=20 m/s, 600 RPM)');
lgd1 = legend('Location','northeast');
lgd1.Title.String = 'Launch Angle';
grid on; xlim([0 inf]); ylim([0 inf]);

subplot(1,2,2);
angles_sweep = linspace(1, 40, 60);
ranges1 = zeros(size(angles_sweep));
for k = 1:length(angles_sweep)
    [~,x,~,~] = simulate(20, angles_sweep(k), 600, 0, params);
    ranges1(k) = x(end);
end
plot(angles_sweep, ranges1, 'b-', 'LineWidth', 2.5);
[~, opt_idx] = max(ranges1);
xline(angles_sweep(opt_idx), 'r--', 'LineWidth', 1.5, ...
      'Label', sprintf('Optimal: %.1f deg', angles_sweep(opt_idx)));
xlabel('Launch Angle (degrees)'); ylabel('Range (m)');
title('Range vs. Launch Angle');
grid on;
sgtitle('Effect of Launch Angle on Frisbee Trajectory', 'FontWeight','bold','FontSize',13);

%% ════════════════════════════════════════════════════════
%  PLOT 2 — Effect of Spin Rate
%% ════════════════════════════════════════════════════════
fprintf('Running Plot 2: Spin Rate Study...\n');
figure('Name','Effect of Spin Rate','NumberTitle','off','Color','w',...
       'Position',[100 100 1200 480]);

subplot(1,2,1);
spins_list = [200 400 600 800 1000 1200];
cmap2 = cool(length(spins_list));
hold on;
for k = 1:length(spins_list)
    [t,x,y,z] = simulate(20, 15, spins_list(k), 0, params);
    plot(x, z, 'Color', cmap2(k,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('%d RPM', spins_list(k)));
end
hold off;
xlabel('Horizontal Distance (m)'); ylabel('Height (m)');
title('Trajectory vs. Spin Rate (v_0=20 m/s, 15 deg)');
lgd2 = legend('Location','northeast');
lgd2.Title.String = 'Spin Rate';
grid on; xlim([0 inf]); ylim([0 inf]);

subplot(1,2,2);
spin_sweep = linspace(100, 1500, 60);
ranges2 = zeros(size(spin_sweep));
for k = 1:length(spin_sweep)
    [~,x,~,~] = simulate(20, 15, spin_sweep(k), 0, params);
    ranges2(k) = x(end);
end
plot(spin_sweep, ranges2, 'm-', 'LineWidth', 2.5);
xlabel('Spin Rate (RPM)'); ylabel('Range (m)');
title('Range vs. Spin Rate');
grid on;
sgtitle('Effect of Spin Rate on Frisbee Flight', 'FontWeight','bold','FontSize',13);

%% ════════════════════════════════════════════════════════
%  PLOT 3 — Effect of Launch Speed
%% ════════════════════════════════════════════════════════
fprintf('Running Plot 3: Launch Speed Study...\n');
figure('Name','Effect of Launch Speed','NumberTitle','off','Color','w',...
       'Position',[100 100 1200 480]);

subplot(1,2,1);
speeds_list = [10 14 18 22 26 30];
cmap3 = winter(length(speeds_list));
hold on;
for k = 1:length(speeds_list)
    [t,x,y,z] = simulate(speeds_list(k), 15, 600, 0, params);
    plot(x, z, 'Color', cmap3(k,:), 'LineWidth', 2, ...
         'DisplayName', sprintf('%d m/s', speeds_list(k)));
end
hold off;
xlabel('Horizontal Distance (m)'); ylabel('Height (m)');
title('Trajectory vs. Launch Speed (15 deg, 600 RPM)');
lgd3 = legend('Location','northeast');
lgd3.Title.String = 'Speed';
grid on; xlim([0 inf]); ylim([0 inf]);

subplot(1,2,2);
speed_sweep = linspace(5, 35, 60);
ranges3 = zeros(size(speed_sweep));
for k = 1:length(speed_sweep)
    [~,x,~,~] = simulate(speed_sweep(k), 15, 600, 0, params);
    ranges3(k) = x(end);
end
plot(speed_sweep, ranges3, 'g-', 'LineWidth', 2.5);
xlabel('Launch Speed (m/s)'); ylabel('Range (m)');
title('Range vs. Launch Speed');
grid on;
sgtitle('Effect of Launch Speed on Frisbee Flight', 'FontWeight','bold','FontSize',13);

%% ════════════════════════════════════════════════════════
%  PLOT 4 — 3D Trajectory
%% ════════════════════════════════════════════════════════
fprintf('Running Plot 4: 3D Trajectory...\n');
figure('Name','3D Trajectory','NumberTitle','off','Color','w',...
       'Position',[100 100 900 600]);

laterals = [-10 -5 0 5 10];
cmap4 = jet(length(laterals));
hold on;
for k = 1:length(laterals)
    [t,x,y,z] = simulate(22, 15, 600, laterals(k), params);
    plot3(x, y, z, 'Color', cmap4(k,:), 'LineWidth', 2, ...
          'DisplayName', sprintf('%d deg', laterals(k)));
end
hold off;
xlabel('X — Downrange (m)'); ylabel('Y — Lateral (m)'); zlabel('Z — Height (m)');
title('3D Frisbee Trajectory - Lateral Launch Angles (v0=22m/s, 15deg, 600RPM)',...
      'FontWeight','bold');
lgd4 = legend('Location','northeast');
lgd4.Title.String = 'Lateral Angle';
grid on; view(30, 20);

%% ════════════════════════════════════════════════════════
%  PLOT 5 — Aerodynamic Polars (Lift, Drag, L/D)
%% ════════════════════════════════════════════════════════
fprintf('Running Plot 5: Aerodynamic Polars...\n');
figure('Name','Aerodynamic Polars','NumberTitle','off','Color','w',...
       'Position',[100 100 1200 480]);

alpha_deg = linspace(-10, 30, 200);
alpha_rad = deg2rad(alpha_deg);
q_dyn = 0.5 * rho * 20^2;
CL_v  = CL0 + CLa .* alpha_rad;
CD_v  = CD0 + CDa .* alpha_rad.^2;
L_v   = q_dyn * A * CL_v;
D_v   = q_dyn * A * CD_v;
LD    = CL_v ./ CD_v;

subplot(1,2,1);
plot(alpha_deg, L_v, 'b-', 'LineWidth', 2.5, 'DisplayName', 'Lift (N)'); hold on;
plot(alpha_deg, D_v, 'r-', 'LineWidth', 2.5, 'DisplayName', 'Drag (N)');
xline(0, 'k--', 'LineWidth', 0.8);
hold off;
xlabel('Angle of Attack (deg)'); ylabel('Force (N)');
title('Lift & Drag vs. Angle of Attack (V=20 m/s)');
legend; grid on;

subplot(1,2,2);
plot(alpha_deg, LD, 'Color', [0 0.5 0.5], 'LineWidth', 2.5);
xline(0, 'k--', 'LineWidth', 0.8);
yline(0, 'k--', 'LineWidth', 0.8);
xlabel('Angle of Attack (deg)'); ylabel('L/D Ratio');
title('Lift-to-Drag Ratio vs. Angle of Attack');
grid on;
sgtitle('Frisbee Aerodynamic Characteristics', 'FontWeight','bold','FontSize',13);

%% ── Print Key Results ───────────────────────────────────
fprintf('\n════════════════════════════════════\n');
fprintf('   KEY SIMULATION RESULTS\n');
fprintf('════════════════════════════════════\n');
[t_ref, x_ref, ~, ~] = simulate(22, 15, 700, 0, params);
fprintf('  Reference flight (22 m/s, 15 deg, 700 RPM):\n');
fprintf('    Range        = %.1f m\n', x_ref(end));
fprintf('    Flight time  = %.2f s\n', t_ref(end));
[~, oi] = max(ranges1);
fprintf('  Optimal launch angle (20 m/s, 600 RPM): %.1f deg\n', angles_sweep(oi));
[~, x_opt, ~, ~] = simulate(20, angles_sweep(oi), 600, 0, params);
fprintf('  Max range at optimal angle: %.1f m\n', x_opt(end));
fprintf('════════════════════════════════════\n');
fprintf('Done! All plots generated.\n');


%% ════════════════════════════════════════════════════════
%  LOCAL FUNCTIONS  (must be at end of script)
%% ════════════════════════════════════════════════════════

function dstate = frisbee_eom(~, state, spin, p)
% Equations of motion for the frisbee
% State: [x, y, z, vx, vy, vz, phi, theta, dphi, dtheta]
    x=state(1); y=state(2); z=state(3);
    vx=state(4); vy=state(5); vz=state(6);
    phi=state(7); theta=state(8);
    dphi=state(9); dtheta=state(10);

    V = sqrt(vx^2 + vy^2 + vz^2);
    if V < 1e-6, V = 1e-6; end

    alpha  = theta;                          % angle of attack
    q_dyn  = 0.5 * p.rho * V^2;

    CL = p.CL0 + p.CLa * alpha;
    CD = p.CD0 + p.CDa * alpha^2;

    Lf = q_dyn * p.A * CL;                  % lift magnitude
    Df = q_dyn * p.A * CD;                  % drag magnitude

    % Lift direction (perpendicular to velocity, upward component)
    vxy = sqrt(vx^2 + vy^2);
    lift_x = -Lf * (vz/V) * cos(phi);
    lift_y = -Lf * (vz/V) * sin(phi);
    lift_z =  Lf * (vxy/V) * cos(phi);

    % Drag (opposes velocity)
    drag_x = -Df * vx/V;
    drag_y = -Df * vy/V;
    drag_z = -Df * vz/V;

    % Translational accelerations
    ax = (lift_x + drag_x) / p.m;
    ay = (lift_y + drag_y) / p.m;
    az = (lift_z + drag_z) / p.m - p.g;

    % Pitching moment
    M_pitch = q_dyn * p.A * (2*p.R) * ...
              (p.Cm0 + p.Cma*alpha + p.Cmq*(dtheta*p.R/V));

    % Rotational equations (with gyroscopic coupling)
    d2phi   = (M_pitch - p.Izz * spin * dtheta) / p.Ixx;
    d2theta = (M_pitch + p.Izz * spin * dphi)   / p.Ixx;

    dstate = [vx; vy; vz; ax; ay; az; dphi; dtheta; d2phi; d2theta];
end

function [value, isterminal, direction] = ground_event(~, state)
% Stop integration when disc hits the ground (z = 0)
    value      = state(3);   % z coordinate
    isterminal = 1;           % stop the integration
    direction  = -1;          % only when z is decreasing
end
