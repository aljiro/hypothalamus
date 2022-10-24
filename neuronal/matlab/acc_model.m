function dr = acc_model( t, r, w, theta )
% Accumbens as a scalar neural network

beta = 1;
f = @(u) 1./(1 + exp(-beta*u));
dr = -r + f(w*r - theta);