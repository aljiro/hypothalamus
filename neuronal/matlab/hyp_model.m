function dr = hyp_model( t, r, insulin, leptin, reward )
% Full hypothalamus model

% Parameters
W_h = 10;
s_h = 0.47;
w_hammel = s_h*W_h;
q_hammel = (1-s_h)*W_h;
w_acc = 8.3;

% Weights
w_acc_app = -0.6; % NAcc -> Approach: Inhibition
w_h_app = 0.6; % Hunger -> approach : excitation
w_avoid_app = 0; % Avoid -> approach : 
w_s_avoid = 0.6; % Satiety -> avoid : excitation
w_app_avoid = 0; % Approach -> avoid : 
w_app_vta = -0.5; % Approach -> vta_gaba: inhibition
w_avoid_vta = 0.5; % Avoid -> vta_gaba: excitation
w_gaba_da = -0.6; % vta_gaba -> vta_da: inhibition
w_orexin_da = 0.5; % Orexin -> vta_da: excitatory
w_leptin_orexin = -0.5; % Leptin -> orexin: excition with more leptin
w_reward = 0.9;

% Currents

% Rate function
beta = 8;
f = @(u) 1./(1 + exp(-beta*u));

beta = 40;
g = @(u) 1./(1 + exp(-beta*u));
% f = @(u)beta*u;

% Previous values
% Hammel rates
r_hammel = r(1:4);

r_sen = r(1);
r_ins = r(2);
r_hunger = r(3);
r_satiety = r(4);

r_approach = r(5);
r_avoid = r(6);
r_vta_gaba = r(7);
r_vta_da = r(8);
r_orexin = r(9);
r_acc_d1 = r(10);

% Map
dr_hammel = hammel_model( t, r_hammel, w_hammel, q_hammel, insulin );
dr_approach = -r_approach + f( w_acc_app*r_acc_d1 + w_h_app*r_hunger + w_avoid_app*r_avoid);
dr_avoid = -r_avoid + f( w_s_avoid*r_satiety + w_app_avoid*r_approach ); % BNST connection missing
dr_vta_gaba = -r_vta_gaba + f( w_app_vta*r_approach + w_avoid_vta*r_avoid  );
dr_vta_da = -r_vta_da + g( w_gaba_da*r_vta_gaba + w_orexin_da*r_orexin );
dr_orexin = -r_orexin + f( w_leptin_orexin*leptin(t) + 0.5 );

% Accumbal subsystem
theta = (r_vta_da - 0.2)*10;
dr_acc_d1 = acc_model( t, r_acc_d1, w_acc, theta ) + + w_reward*reward(t);

dr = [dr_hammel;
      dr_approach;
      dr_avoid;
      dr_vta_gaba;
      dr_vta_da;
      dr_orexin;
      dr_acc_d1];
end