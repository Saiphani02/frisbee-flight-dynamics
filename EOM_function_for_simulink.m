%% =========================================================
%  PASTE THIS CODE INTO THE "Frisbee_EOM" MATLAB FUNCTION
%  BLOCK IN SIMULINK (if it is empty after running the builder)
%
%  HOW:
%  1. Double-click the Frisbee_EOM block in Simulink
%  2. Select all existing text (Ctrl+A) and delete it
%  3. Copy everything below this comment and paste it in
%  4. Click Save (Ctrl+S) and close the editor
% =========================================================

function dstate = Frisbee_EOM(u)
%% Frisbee Equations of Motion
%  Input  u      : [state(10); spin] — 11 elements
%  Output dstate : [dx;dy;dz;dvx;dvy;dvz;dphi;dtheta;d2phi;d2theta]

% Unpack state
vx     = u(4);   vy    = u(5);   vz     = u(6);
phi    = u(7);   theta = u(8);
dphi   = u(9);   dtheta= u(10);
spin   = u(11);

% Physical parameters
m   = 0.175;  R = 0.107;  A = pi*R^2;  rho = 1.225;  g = 9.81;
CL0 =  0.10;  CLa =  1.80;
CD0 =  0.08;  CDa =  2.72;
Cm0 =  0.01;  Cma = -0.40;  Cmq = -0.005;
Izz = 0.5*m*R^2;  Ixx = 0.25*m*R^2;

% Airspeed
V = sqrt(vx^2 + vy^2 + vz^2);
if V < 1e-6,  V = 1e-6;  end

% Aerodynamic forces
alpha = theta;
q_dyn = 0.5 * rho * V^2;
CL    = CL0 + CLa * alpha;
CD    = CD0 + CDa * alpha^2;
Lf    = q_dyn * A * CL;
Df    = q_dyn * A * CD;

% Force components
vxy    = sqrt(vx^2 + vy^2);
lift_x = -Lf * (vz/V) * cos(phi);
lift_y = -Lf * (vz/V) * sin(phi);
lift_z =  Lf * (vxy/V) * cos(phi);
drag_x = -Df * vx/V;
drag_y = -Df * vy/V;
drag_z = -Df * vz/V;

% Translational accelerations
ax = (lift_x + drag_x) / m;
ay = (lift_y + drag_y) / m;
az = (lift_z + drag_z) / m - g;

% Pitching moment + rotational accelerations
M_pitch = q_dyn*A*(2*R)*(Cm0 + Cma*alpha + Cmq*(dtheta*R/V));
d2phi   = (M_pitch - Izz*spin*dtheta) / Ixx;
d2theta = (M_pitch + Izz*spin*dphi  ) / Ixx;

% Output derivative vector
dstate = [vx; vy; vz; ax; ay; az; dphi; dtheta; d2phi; d2theta];
end
