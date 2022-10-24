function dr = temperature_sim( t, r, Ta, Tp, Gf, Gm, Gc, Gb, alpha, k1, k2, k3 )
% Simulation of an agent performing a simple task

% Parameters
s = 0.1;

% Position
global agent
cp = [agent.x(end), agent.y(end)];
% Weights

% Rates
Ts = r(1);
Tb = r(2);
r_model = r(3:12);
r_da = r(10);

% Changing movement

a = 0.1;
dir = (Tp - Ts);
dir = dir/norm(dir);
dp = (rand(1,2)-0.5 + a*dir)*r_da*s;
p = cp + dp;
agent.x = [agent.x, p(1)];
agent.y = [agent.y, p(2)];


n = 1;
E = n^(-0.25);
dTs = alpha(Tb)*(Tb - Ts)*k1 + E*(Ta(p(1), p(2)) - Ts)*k2;
dTb = -alpha(Tb)*(Tb - Ts)*k3 + Gb + Gf(Tb) + Gm(Tb) + Gc(Tb);
agent.fat = agent.fat - 0.0001*Gf(Tb);
leptin = @(t) agent.fat;
skin_temp = @(t) abs(Tb - Tp)/100;
reward = @(t) 0;

dr_model = hyp_model( t, r_model, skin_temp, leptin, reward );

dr = [dTs;
      dTb;
      dr_model];
     



