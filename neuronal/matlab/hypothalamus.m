% Hammel rate model
close all
clc
y0 = [0, 0.7, 0, 0];
T = 25;
W = 10;
s = 0.47;

k_ins = 0.07;
I_r = @(t) (t > 10).*(k_ins*10-k_ins*(t-10)) + (t <= 10).*(k_ins*t);
I_ins = @(t) (I_r(t)>0).*I_r(t);
[t,y] = ode45( @(t,r)hammel_model(t,r,s*W,(1-s)*W,I_ins), [0 T], y0 );
figure('pos', [0,0,600,1000])
subplot 411
plot( t', I_ins(t), 'linewidth', 2 )
title('Insulin level')


subplot 412
plot( t', y(:,1), 'displayname', 'Sensitive', 'linewidth', 2 )
hold on
plot( t', y(:,2), 'displayname', 'Insensitive', 'linewidth', 2  )
legend
title('Energy sensitive/insensitive neurons')
ylabel('Firing rate')
subplot 413
plot( t', y(:,3), 'linewidth', 2 )
ylabel('Firing rate')
axis([0,T,0,2])
title('Hunger neurons')

subplot 414
plot( t', y(:,4), 'linewidth', 2 )
axis([0,T,0,2])
ylabel('Firing rate')
title('Satiety neurons')
xlabel('time')

%% Hammel model parametric search

close all
clc

T = 30;
N = 50;
M = 100;
tt = linspace(0,T,M);
Yh = zeros(N,M);
Ys = zeros(N,M);
s = linspace(0,1,N);

k_ins = 0.07;
I_r = @(t) (t > 10).*(k_ins*10-k_ins*(t-10)) + (t <= 10).*(k_ins*t);
I_ins = @(t) (I_r(t)>0).*I_r(t);

W = 10;
NIns = 4;
rins = linspace(0.2,1,NIns);
figure('pos', [0,0,500,1000])
for j = 1:NIns

for i = 1:N
    y0 = [0, rins(j), 0, 0];
    [t,y] = ode45( @(t,r)hammel_model(t,r,s(i)*W,(1-s(i))*W, I_ins), tt, y0 );
    Yh( i,: ) = y(:,3)';
    Ys( i,: ) = y(:,4)';
end

subplot(NIns, 2, 2*j - 1)

imagesc(Yh)
title(sprintf('Hunger r_{ins} = %.2f', rins(j)))
xlabel('time')
ylabel('s')

subplot (NIns,2, 2*j)
imagesc(Ys)
title(sprintf('Satiety r_{ins} = %.2f', rins(j)))
xlabel('time')
ylabel('s')

end

%% Accumbal model model

close all
clc
y0 = 0;
T = 200;

Nw = 100;
Ntheta = 100;
w = linspace(-20, 20, Nw );
theta = linspace(-20, 20, Ntheta);
S = zeros( Nw, Ntheta );

for i = 1:Nw
    for j = 1:Ntheta
        [t,y] = ode45( @(t,r)acc_model(t, r, w(i), theta(j)), [0 T], y0 );        
        S(i,j) = y(end);
    end
end

figure('pos', [0,0,1000,500]) 
subplot 121
imagesc( S )
colorbar
xlabel( '\theta' );
ylabel( 'w' );
title('Bifurcation diagram accumbens')
set( gca, 'YDir', 'Normal' )
idxs_w = 1:10:Nw;
idxs_t = 1:10:Ntheta;
set( gca, 'YTick', idxs_w )
set( gca, 'XTick', idxs_t )
set( gca, 'YTickLabels', ...
          arrayfun(@(i)sprintf('%.1f',w(i)), ...
          idxs_w, 'UniformOutput', false ) )
set( gca, 'XTickLabels', ...
          arrayfun(@(i)sprintf('%.1f',theta(i)), ...
          idxs_t, 'UniformOutput', false ) )

subplot 122
surf( w, theta, S )
title('Catastrophe surface')
xlabel('\theta')
ylabel('w')
zlabel('u')

%% Full model
close all
clc

T = 25;
k_ins = 0.07;
I_r = @(t) (t > 10).*(k_ins*10-k_ins*(t-10)) + (t <= 10).*(k_ins*t);
I_ins = @(t) (I_r(t)>0).*I_r(t);

I_lep_low = @(t)0.1*ones(size(t));
I_lep_high = @(t) 1.0*ones(size(t));

% No reward
% reward = @(t)zeros(t); 
% Reward delivered
% reward = @(t) (t > 10) && (t < 11);
reward = @(t)0;

% Low energy
% I_lep = @(t)I_lep_low(t);
% High energy
I_lep = @(t)I_lep_high(t);
% 
r0_hammel = [0, 0.7, 0, 0];
r0_LH = [0, 0];
r0_vta = [0.5, 0];
r0_orex = 0;
r0_acc = 0.5;

% Model simulation
y0 = [r0_hammel, r0_LH, r0_vta, r0_orex, r0_acc];
[t,y] = ode45( @(t,r)hyp_model(t,r,I_ins, I_lep, reward), [0 T], y0 );

% Plots
figure('pos', [0,0,600,1000])
subplot 411
% Hormones
plot( t, I_ins(t), 'linewidth', 2, 'DisplayName', 'Insulin' )
hold on
plot( t, I_lep(t), 'linewidth', 2, 'DisplayName', 'Leptin' )
ylabel('Concentration')
legend
title('Hormone level')

subplot 412
% Approach/Avoid pathways
plot( t', y(:,5), 'displayname', 'Approach', 'linewidth', 2 )
hold on
plot( t', y(:,6), 'displayname', 'Avoid', 'linewidth', 2  )
legend
title('LH channels')
ylabel('Firing rate')

subplot 413
% Accumbal neurons
plot( t', y(:,10), 'displayname', 'NAcc', 'linewidth', 2 )

title('Accumbal modulation')
ylabel('Firing rate')

subplot 414
% Dopamine system
plot( t', y(:,7), 'displayname', 'DA_GABA', 'linewidth', 2 )
hold on
plot( t', y(:,8), 'displayname', 'DA', 'linewidth', 2  )
plot( t', y(:,9), 'displayname', 'Orexin', 'linewidth', 2  )
legend
title('Dopamine system')
ylabel('Firing rate')
xlabel('Time')


%% Agent simulation

close all
clc

ins0 = 0.8;

I_lep_low = @(t)0.1*ones(size(t));
I_lep_high = @(t) 1.0*ones(size(t));

% Low energy
I_lep = @(t)I_lep_low(t);
% High energy
% I_lep = @(t)I_lep_high(t);

T = 25;
t = linspace(0,T,200);
w_a0 = 0.5;
w_b0 = 0.5;
r_a0 = 0;
r_b0 = 0;
r_tan0 = 1;

r0_hammel = [0, 0.7, 0, 0];
r0_LH = [0, 0];
r0_vta = [0.5, 0];
r0_orex = 0;
r0_acc = 0.5;

clear agent
global agent
agent = struct('x', [0], 'y', [0], 'insulin', ins0, 'tr', 0);

y_model0 = [r0_hammel, r0_LH, r0_vta, r0_orex, r0_acc];
y_agent0 = [w_a0, w_b0, r_a0, r_b0, r_tan0];
y0 = [y_agent0, y_model0];

[t,y] = ode45( @(t,r)agent_sim(t, r, ins0, I_lep), t, y0 );

figure
subplot 311
plot( t, y(:,12), 'linewidth', 2, 'displayname', 'VTA GABA')
hold on
plot( t, y(:,5), 'linewidth', 2, 'displayname', 'TAN')
% plot( t, y(:,end), 'linewidth', 2, 'displayname', 'NAcc')
legend
axis([0,T,0 1] )

subplot 312
plot( t, y(:,3), 'linewidth', 2, 'DisplayName', 'Point a' )
hold on
plot( t, y(:,4), 'linewidth', 2, 'DisplayName', 'Point b' )
legend

subplot 313
plot( t, y(:,1), 'linewidth', 2, 'DisplayName', 'Weight a' )
hold on
plot( t, y(:,2), 'linewidth', 2, 'DisplayName', 'Weight b' )
axis([0, T, 0 1])
legend


figure
% for i = 20:length(agent.x)-1
% plot( agent.x(i-19:i), agent.y(i-19:i), 'k' )
% hold on
% plot( agent.x(i), agent.y(i), 'k.', 'markersize', 10 )
% plot( -1, 1, 'r*', 'markersize', 20 )
% plot( 1, 1, 'b*', 'markersize', 20 )
% hold off
% axis( [-2 2 -1 2] );
% pause(agent.t(i+1)-agent.t(i))
% end


plot( agent.x, agent.y, 'k' )
hold on
plot( -1, 1, 'r*', 'markersize', 20, 'linewidth', 2 )
plot( -1, 1, 'ro', 'markersize', 25, 'linewidth', 2 )
plot( 1, 1, 'b*', 'markersize', 20, 'linewidth', 2 )
axis( [-2 2 -1 2] );
title('Agent trajectory')

%% Agent simulation learning

close all
clc

ins0 = 1;

I_lep_low = @(t)0.01*ones(size(t));
I_lep_high = @(t) 1*ones(size(t));

% Low energy
I_lep = @(t)I_lep_low(t);
% High energy
% I_lep = @(t)I_lep_high(t);

T = 25;
t = linspace(0,T,200);
num_trials = 50;

ax = -1.5;
bx = 1.5;
ay = -0.5;
by = 1.5;
ns = 100;
dx = (bx - ax)/ns;
dy = (by - ay)/ns;
X = zeros( ns, ns );
wa_n = zeros(1,num_trials);
wb_n = zeros(1,num_trials);

wa_n(1) = 0.5;
wb_n(1) = 0.5;
histos = zeros(1,30);
edges = linspace(ax, bx, 31); 

for i = 1:num_trials
    w_a0 = wa_n(i);
    w_b0 = wb_n(i);
    r_a0 = 0;
    r_b0 = 0;
    r_tan0 = 1;

    r0_hammel = [0, 0.7, 0, 0];
    r0_LH = [0, 0];
    r0_vta = [0.5, 0];
    r0_orex = 0;
    r0_acc = 0.5;

    clear agent
    global agent
    agent = struct('x', [0], 'y', [0], 'insulin', ins0, 'tr', 0);

    y_model0 = [r0_hammel, r0_LH, r0_vta, r0_orex, r0_acc];
    y_agent0 = [w_a0, w_b0, r_a0, r_b0, r_tan0];
    y0 = [y_agent0, y_model0];

    [t,y] = ode45( @(t,r)agent_sim(t, r, ins0, I_lep), t, y0 );
    
    for j = 1:length(agent.x)
        px = ceil((agent.x(j) - ax)/dx);
        py = ceil((agent.y(j) - ay)/dy);
        X(py,px) = X(py,px) + 1;
    end
    
    [h, edges] = histcounts( agent.x, edges );
    histos = histos + h;
    
    fprintf('Iteration %d\n', i)
    w = [y(end,1), y(end,2)];
    w = w/norm(w);
    wa_n(i+1) = w(1);
    wb_n(i+1) = w(2);
end

figure('pos', [0 0 900 400])
histos = histos/num_trials;
subplot 121
X = X/num_trials;
X(X>50) = 50;
imagesc(X)
set(gca, 'YDir', 'normal')
title('Time spend at each point')
set(gca, 'XTick', [1, 50, 100], 'XTickLabels', [-1.5, 0, 1.5])
set(gca, 'YTick', [])
xlabel('x')
ylabel('y')

axes('Position',[.13 .79 .335 .14])
box on
bar((edges(1:end-1)+edges(2:end))/2, histos)
set(gca, 'XTick', [], 'YTick', [])

subplot 122
plot( wa_n, 'linewidth', 2, 'DisplayName', 'Weight a' )
hold on
plot( wb_n, 'linewidth', 2, 'DisplayName', 'Weight b' )
ylabel('Synaptic strength')
xlabel('time')
title('Evolution of synaptic weights')
legend


%% Temperature simulation
close all
clc
T = 50;
t = linspace(0, T, 200);
Mt = 1;
MaxT = 2;
Tp = 36;
Ta = @(x,y) 15*x + 25;

piecewise = @(x,l,u) min(max((x-l)/(u-l),0),1);

q = 2;   
M = 50;
Gb = 21.66*M^(-0.25);
C = 4.32*M^(-0.426);
GmMax = (q/(q+1))*(Tp*C-Gb);       
GfMax = (1./(q+1))*(Tp*C-Gb);       
GcMax = (0.5)*(Tp*C-Gb);

TU = 35;
k1 = 0.5;
k3plus = 1;
k3 = Gb/(Tp-TU)+k3plus;
k2 = k1/(k3*(Tp-TU)/Gb-1);
AlMin = 1./(k3/C - k1/k2);

[fl,fu] = deal(Tp-0.3,Tp-0.5);   
[ml,mu] = deal(Tp-0.5,Tp-0.7);   
[cl,cu] = deal(Tp+0.3, Tp+5.0);
[al,au] = deal(Tp-0.1, Tp+0.1);
Gf = @(Tb)GfMax*arrayfun( @(x)piecewise(x,fl,fu), Tb );
Gm = @(Tb)GmMax*arrayfun( @(x)piecewise(x,ml,mu), Tb );
Gc = @(Tb)GcMax*arrayfun( @(x)piecewise(x,cl,cu), Tb );
alpha = @(Tb)AlMin+(1.-AlMin)*arrayfun( @(x)piecewise(x,al,au), Tb);

clear agent
global agent
agent = struct('x', [2*rand-1], 'y', [rand-0.5], 'fat', 1);

r0_hammel = [0, Tp/100, 0, 0];
r0_LH = [0, 0];
r0_vta = [0.5, 0];
r0_orex = 0.5;
r0_acc = 0.5;
Ts_0 = Ta(agent.x, agent.y); 
Tb_0 = Ta(agent.x, agent.y);

y_model0 = [r0_hammel, r0_LH, r0_vta, r0_orex, r0_acc];
y_agent0 = [Ts_0, Tb_0];
y0 = [y_agent0, y_model0];

[t,y] = ode45( @(t,r)temperature_sim(t, r, Ta, Tp, Gf, Gm, Gc, Gb, alpha, k1, k2, k3), t, y0 );

figure
subplot 311
plot( t, y(:,1), 'linewidth', 2, 'DisplayName', 'T Skin' )
hold on
plot( t, y(:,2), 'linewidth', 2, 'DisplayName', 'T Body' )
legend

subplot 312
plot( t, y(:,5), 'linewidth', 2, 'DisplayName', 'H' )
hold on
plot( t, y(:,6), 'linewidth', 2, 'DisplayName', 'S' )
legend
subplot 313
plot( t, y(:,10), 'linewidth', 2, 'DisplayName', 'Dopamine response' )
hold on
plot( t, y(:,11), 'linewidth', 2, 'DisplayName', 'Energy budget' )
legend
figure
plot( agent.x, agent.y, 'k', 'linewidth', 1 )
hold on
plot( agent.x(1), agent.y(1), 'b.', 'markersize',30 )
plot( agent.x(end), agent.y(end), 'r.', 'markersize', 30 )
grid on
colormap(flipud(hot(16)))
cb = colorbar('SouthOutside');
tb = linspace(10, 40, 8);

set(cb, 'XTick', linspace(0,1,8), 'XTickLabel', arrayfun(@(x)sprintf('%.1f', x), tb, 'uniformoutput', false))
axis([-Mt Mt min(agent.y)-1 max(agent.y)+1])
xticks(gca, [])
yticks(gca,[])
set( gca, 'pos', [0.13,0.13,0.775,0.55])

axes('Position',[.13 .7 .775 .16])
box on
tb = linspace(-1,1,1000)*2+Tp;
plot( tb, Gf(tb), 'linewidth', 2 )
hold on
plot( tb, Gm(tb),'linewidth', 2 )
plot( tb, Gc(tb),'linewidth', 2 )
plot( tb, alpha(tb),'linewidth', 2 )
set(gca, 'XTick', [], 'YTick', [])

axes('Position',[.13 .88 .775 .10])
edges = linspace(-Mt, Mt, 101);
histogram( agent.x, edges )
set(gca, 'XTick', [], 'YTick', [])

Ta(agent.x(end), agent.y(end))