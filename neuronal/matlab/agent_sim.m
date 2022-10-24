function dr = agent_sim( t, r, ins0, leptin )
% Simulation of an agent performing a simple task

% Parameters
s = 0.1;
a = [1,1];
b = [-1,1];
% Position
global agent
cp = [agent.x(end), agent.y(end)];
% Weights
w_a = r(1);
w_b = r(2);
w_ba = -0.9;
w_ab = -0.9;
w_gaba_tan = -0.9;

% Rates
r_a = r(3);
r_b = r(4);
r_tan = r(5);

r_model = r(6:15);
r_da = r(13);
r_vta_gaba = r(12);

% Functions
beta = 8;
f = @(u) 1./(1 + exp(-beta*u));
reward = @(t) 0;

% Changing movement
dira = (a - cp);
dira = dira/norm(dira);
dirb = (b - cp);
dirb = dirb/norm(dirb);
dp = ((rand(1,2)-0.5) + dira*r_a + dirb*r_b)*r_da*s;
p = cp + dp;
agent.x = [agent.x, p(1)];
agent.y = [agent.y, p(2)];

% Maps
alpha = 0.1;
dw_a = alpha*abs(heaviside(0.5-r_tan))*r_a*r_da;
dw_b = alpha*abs(heaviside(0.5-r_tan))*r_b*r_da;
d_tan = -r_tan + f(w_gaba_tan*(r_vta_gaba) + 0.7);
dr_a = -r_a + f( w_a*r_da + w_ba*r_b ) + rand*0.1;
dr_b = -r_b + f( w_b*r_da + w_ab*r_a ) + rand*0.1;

if norm(cp - b)< .1
    reward = @(t)1;
    agent.tr = t;
    agent.insulin = ins0*exp(-0.1*(agent.tr - t));
end

insulin = @(t) agent.insulin;

dr_model = hyp_model( t, r_model, insulin, leptin, reward );

dr = [dw_a;
     dw_b;
     dr_a;
     dr_b;
     d_tan;
     dr_model];
     



