function dr = hyp_model2( t, r, hormone, leptin, reward, punish )
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
w_avoid_da2 = 0.5;
w_gaba_da = -0.6; % vta_gaba -> vta_da: inhibition
w_orexin_da = 0.5; % Orexin -> vta_da: excitatory
w_leptin_orexin = -0.5; % Leptin -> orexin: excition with more leptin
w_lat_st = -10;
w_med_st = -10;
w_bnst_avoid = 0.9; % Excitatory input of punishments
w_punish = 1;
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

% LH
r_approach = r(5);
r_avoid = r(6);
r_orexin = r(7);
% VTA
r_vta_gaba = r(8);
r_vta_da1 = r(9);
r_vta_da2 = r(10);
% Accumbens
r_acc_lat = r(11);
r_acc_med = r(12);
% Striatum
r_st1 = r(13);
r_st2 = r(14);
% Amygdala
r_bnst = r(15);

% Maps
% -----------------------
% LH
dr_hammel = hammel_model( t, r_hammel, w_hammel, q_hammel, hormone );
dr_approach = -r_approach + f( w_acc_app*r_acc_lat + w_h_app*r_hunger + w_avoid_app*r_avoid);
dr_avoid = -r_avoid + f( w_s_avoid*r_satiety + w_app_avoid*r_approach + w_bnst_avoid*r_bnst );
dr_orexin = -r_orexin + f( w_leptin_orexin*leptin + 0.5 );

% VTA
dr_vta_gaba = -r_vta_gaba + f( w_app_vta*r_approach + w_avoid_vta*r_avoid  );
dr_vta_da1 = -r_vta_da1 + g( w_gaba_da*r_vta_gaba + w_orexin_da*r_orexin );
dr_vta_da2 = -r_vta_da2 + g( w_avoid_da2*r_avoid );

% Accumbal subsystem
theta1 = (r_vta_da1 - 0.2)*10;
theta2 = (r_vta_da2 - 0.2)*10;
dr_acc_lat = acc_model( t, r_acc_lat, w_acc, theta1 );
dr_acc_med = acc_model( t, r_acc_med, w_acc, theta2 );

% Str
tonic = 0.1;
dr_st1 = -r_st1 + f( w_lat_st*r_acc_lat + tonic );
dr_st2 = -r_st2 + f( w_med_st*r_acc_med + tonic );

% Amygdala
dr_bnst = -r_bnst + f( w_punish*punish );
% 
% dr_hammel
%       dr_approach
%       dr_avoid
%       dr_orexin
%       dr_vta_gaba
%       dr_vta_da1
%       dr_vta_da2
%       dr_acc_lat
%       dr_acc_med
%       dr_st1
%       dr_st2
%       dr_bnst
%       
dr = [dr_hammel;
      dr_approach;
      dr_avoid;
      dr_orexin;
      dr_vta_gaba;
      dr_vta_da1;
      dr_vta_da2;
      dr_acc_lat;
      dr_acc_med;
      dr_st1;
      dr_st2;
      dr_bnst];
end