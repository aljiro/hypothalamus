function dr = hammel_model( t, r, we, wi, insulin )
% Hammel mechanism with rate neurons
% r(1) : Insulin sensitive neurons
% r(2) : Insulin insensitive
% r(3) : Hunger neurons
% r(4) : Satiety neurons

% Parameters
% Time constants
tau_sen = 1;
tau_ins = 1;
tau_h = 1;
tau_s = .5;

% Rates
k_ins = 0; % Constant, mostly

% Weights
w_energy = 0.5;
w_sen_h = we;
w_ins_h = -wi;
w_sen_s = -wi;
w_ins_s = we;

% External input

% Rate function
beta = 8;
f = @(u) 1./(1 + exp(-beta*u));
g = @(u)beta*u;

% Previous values
r_sen = r(1);
r_ins = r(2);
r_h = r(3);
r_s = r(4);

% Map
dr_sen = -r_sen/tau_sen + g( w_energy*insulin )/tau_sen;
dr_ins = k_ins/tau_ins;
dr_h = -r_h/tau_h + f( w_sen_h*r_sen + w_ins_h*r_ins )/tau_h;
dr_s = -r_s/tau_s + f( w_sen_s*r_sen + w_ins_s*r_ins )/tau_s;

dr = [dr_sen;
      dr_ins;
      dr_h;
      dr_s];