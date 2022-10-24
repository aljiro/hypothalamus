%% Bifurcations
close all

a = -5;
b = 5;
beta = 1;
f = @(u) 1./(1 + exp(-beta*u));
k = 1;
u = linspace(a, b, 100 );

subplot 131
w = -9;
theta = -2;
plot( u, f(w*u - theta), 'linewidth', 2 )
hold on
plot( u, k*u, 'linewidth', 2 )
axis([a,b,-1 2])

subplot 132


subplot 133