% Experiment 2
% ----------------------------------------------------------
% Approach and avoidance
close all
clear all
clc

T = 2;
sim_time = linspace( 0, T, 100 );

% Initial conditions
r0_hammel = [0, 0.7, 0, 0];
r0_LH = [0, 0, 0];
r0_vta = [0.5, 0, 0];
r0_acc = [0.0, 0];
r0_st = [0.0, 0.0];
r0_bnst = [0];
y0 = [r0_hammel, r0_LH, r0_vta, r0_acc, r0_st, r0_bnst];

reward = 0;
punish = 0;

objects = [-1,1;
            1,1];
punishments = [1, 0];
rewards = [0, 0];
hormone = 100.0;
leptin = 0.0;
% Bot data structure
bot = struct('pos', [0,0],...
             'sensor_left', [-0.1, 0.1],...
             'sensor_right', [0.1, 0.1],...
             'vel', [0,1]);
% Time step
h = 0.1;
% Shape of the robot
shape = [-0.1, -0.1;
         -0.1, 0.1;
         0.1, 0.1;
         0.1, -0.1];

reward = 0;
punish = 0;

st1_ts = zeros(size(sim_time));
st2_ts = zeros(size(sim_time));
da_lat_ts = zeros(size(sim_time));
da_med_ts = zeros(size(sim_time));
acc_lat_ts = zeros(size(sim_time));
acc_med_ts = zeros(size(sim_time));
app_ts = zeros(size(sim_time));
avoid_ts = zeros(size(sim_time));

figure('pos', [0, 100, 600, 500])
ax1 = axes('Position', [0.025,0.3,0.95,0.7]);
ax2 = axes('Position', [0.025,0.025,0.45,0.27]);
ax3 = axes('Position', [0.51,0.025,0.45,0.27]);

for i = 1:length(sim_time)-1
%     punish = 10;
    I = [sim_time(i),sim_time(i+1)];
    f = @(t, y)hyp_model2( t, y, hormone, leptin, reward, punish );
    [t,y] = ode45( f, I, y0 );

    % Determine sensor input
    sensor1 = exp(-(norm( bot.pos + bot.sensor_left - objects(1,:)) + ...
              norm( bot.pos + bot.sensor_left - objects(2,:))));
    sensor2 = exp(-(norm( bot.pos + bot.sensor_right - objects(1,:)) + ...
              norm( bot.pos + bot.sensor_right - objects(2,:))));
          
    action1 = 1 - y(13,end);
    action2 = 1 - y(14,end);
    
    % Define motor output
    vel_left = action1*sensor1 + action2*sensor2 + randn*0.1;
    vel_right = action1*sensor2 + action2*sensor1 + randn*0.1;
    
    diff = (vel_left - vel_right)/2;
    theta = diff*pi/4;
    
    R = [cos(theta) -sin(theta);
         sin(theta) cos(theta)];
    vel = bot.vel/norm(bot.vel);
   
    vel = ((R*vel')*(vel_left + vel_right))';
    bot.vel = vel;
    norm(bot.pos, 1)
    if norm(bot.pos, 1) > 2
        bot.vel = -bot.vel;
    end
    
    bot.pos = bot.pos + h*bot.vel;
    shape = (R*shape')';
    
    d_1 = norm( bot.pos - objects(1,:));
    d_2 = norm( bot.pos - objects(2,:));
    
    if d_1 < 0.1 
        reward = rewards(1);
        punish = punishments(1);
    end
    
    if d_2 < 0.1 
        reward = rewards(2);
        punish = punishments(2);
    end   
    
    st1_ts(i) = y(end, 13);
    st2_ts(i) = y(end, 14);
    da_lat_ts(i) =  y(end,9);
    da_med_ts(i) = y(end, 10);
    acc_lat_ts(i) =  y(end, 11);
    acc_med_ts(i) = y(end, 12);
    app_ts(i) =  y(end, 5);
    avoid_ts(i) = y(end, 6);
    % Drawing
    axes(ax1)
    hold off
    plot( objects(1,1), objects(1,2), 'g.', 'markersize', 35 );
    hold on
    plot( objects(2,1), objects(2,2), 'r.', 'markersize', 35 );
    pgon = polyshape(shape(:,1) + bot.pos(1), shape(:,2) + bot.pos(2));
    plot(pgon)
    axis([-2 2 -2 2])
    xticks([])
    yticks([])
    axis equal
    
    axes(ax2)
    hold off
    plot( sim_time, st1_ts, 'linewidth', 2, 'DisplayName', 'st1' )
    hold on
    plot( sim_time, st2_ts, 'linewidth', 2, 'DisplayName', 'st2' )
    legend
    
    axes(ax3)
%     hold off
%     plot( sim_time, acc_lat_ts, 'linewidth', 2, 'DisplayName', 'Acc lat' )
%     hold on
%     plot( sim_time, acc_med_ts, 'linewidth', 2, 'DisplayName', 'Acc med' )
    
    hold off
    plot( sim_time, app_ts, 'linewidth', 2, 'DisplayName', 'Approach' )
    hold on
    plot( sim_time, avoid_ts, 'linewidth', 2, 'DisplayName', 'Avoid' )
    
%     hold off
%     plot( sim_time, da_lat_ts, 'linewidth', 2, 'DisplayName', 'DA lat' )
%     hold on
%     plot( sim_time, da_med_ts, 'linewidth', 2, 'DisplayName', 'DA med' )
    legend
    
    pause(0.01)
    
end