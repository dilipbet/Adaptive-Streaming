function[cap_mat,cap_bs,zz,active_links,zz_bs]=physical_layer(z_h,z_u,macro_side,cell_side,helpers_per_cell,...
    users_per_cell,W_lte,W_wifi,freq_reuse,T)
%*** TAKES VARIOUS SYSTEM PARAMETERS AS INPUT AND SPITS OUT 
% 1) THE CAPACITIES ACHIEVABLE FROM HELPERS TO USERS BY TREATING 
% INTERFERENCE AS NOISE.
% 2) THE CAPACITIES ACHIELVABLE FROM BS TO USERS
% 3) THE CONNECTIVITY GRAPH INCIDENCE MATRIX ASSUMING THAT EACH HELPER
% SERVES USERS ONLY WITHIN A CERTAIN RADIUS *******************
%*******************OFDM SPECS***************************
% 1 resource element = 1 OFDM symbol*1 subcarrier
% 1 resource block = 7 OFDM symbols* 12 sub-carriers
% 7 OFDM symbols = 0.5 milliseconds = minimum possible coherence time that
% can occur in any system
% 12 subcarriers = 12*15 = 180KHz = minimum possible coherence bandwidth =>
% delay spread = 5.5 micsoseconds (approx).
% We assume that that the coherence time of our urban system is 10 ms (can also
% assume 100 ms if the environment is pedestrian).
% fading coefficient will be same for (coherence time*coherence bandwidth)
% slots
% in our case coherence time = 10ms = 20*0.5ms = 20*7 = 140 symbols
OFDM_symbols = 7;
num_subcarriers = 12;
resource_block_duration = 0.5*10^(-3); %in seconds
coherence_time = (0.5)*10^(-3); % in seconds
symbols_same_fade_time = (coherence_time/resource_block_duration)*OFDM_symbols;
symbols_same_fade_freq = num_subcarriers; %number of subcarriers
symbols_same_fade = symbols_same_fade_time*symbols_same_fade_freq;%=12*7=84
chunk_duration = 2/3; %in seconds
schedule_slot = chunk_duration;
system_bandwidth = 18*10^6; %20MHz
subcarrier_bandwidth = 15*10^3;%15 KHz
num_blocks_freq = system_bandwidth/(num_subcarriers*subcarrier_bandwidth);
num_blocks_time = floor(schedule_slot/coherence_time);
num_blocks1 = num_blocks_time*num_blocks_freq;
total_symbols = symbols_same_fade*num_blocks1;
num_blocks = 10^4;


delta = 40; %pathloss parameter
alpha = 3.5; %pathloss parameter
noise_power = 1;
%macro_side = 400; % side length of macro square grid
%cell_side = 80; % side length of cell square
num_cells = (macro_side/cell_side)^2; %total number of cells
%helpers_per_cell = 2;
%users_per_cell = 2;
num_helpers = num_cells*helpers_per_cell;
num_users = num_cells*users_per_cell;
x_min = -floor(macro_side/(2*cell_side));% minimum poosible x-coordinate of center of a cell
y_min = x_min; %min possible y-coordinate of center of a cell
x_max = -x_min; %max possible x-coordinate of centre of cell
y_max = x_max;
femto_power = 10^(2);% helper power = 23dBm
BS_power = 10^(4); %BS power =  43 dBm
helper_power = femto_power*ones(num_cells*helpers_per_cell,1);
helper_power(1:num_helpers) = 0;
%if(i>50 && i<=100)
%    helper_power(2) = 0;
%end
%helper_power(3:num_cells*helpers_per_cell,1)=0;
helper_user_distance = abs(repmat(z_h,1,length(z_u)) -  repmat(z_u,length(z_h),1));
bs_user_distance = abs(z_u);
zz = (helper_user_distance <= 60); %connectivity matrix
zz_bs = zeros(1,num_users);
zz_bs(1:4) = 1;%BS offers service only to 4 users
zz(3:25,:)=0; %only helpers 1 and 2 are switched ON and active
zz(1:2,5:50)=0; % only users 1:4 are active and requesting
zz1 = ones(num_helpers,num_users);
zz1(1:2,5:50)=0;
zz1(3:25,1:50) = 0;
path_loss = 1./(1+(helper_user_distance/delta).^alpha);
path_loss_bs = 1./(1+(bs_user_distance/delta).^alpha);
helper_power = repmat(helper_power,1,num_users);
helper_user_strength = (helper_power).*path_loss.*zz1;
bs_user_strength = BS_power*path_loss_bs.*zz_bs;
bs_user_strength_rep = repmat(bs_user_strength,num_blocks,1);
fade_bs = complex(sqrt(1/2)*randn(num_blocks,num_users),...
    sqrt(1/2)*randn(num_blocks,num_users));
bs_user_strength_iid = bs_user_strength_rep.*(abs(fade_bs).^2);
%***** Until this point, we have calculated the signal power from every
%helper to every user in the whole system. But, it is important that
%helpers transmit to only those users which are close enough. It is silly
%to schedule a helper tranmission to a user who is at the other end of the
%campus. Thus, we need the connectivity matrix or the graph incidence
%matrix ZZ. Defining ZZ is important. We can also generalize ZZ to include
%only those heleprs which have the requested file and also the frequency reuse
%Thus, ZZ is a very important matrix. 

% How to define ZZ?
% We assume that a helper will schedule its transmissions to anybody within
% 100m radius.

helper_user_strength_repeat = repmat(helper_user_strength,[1,1,num_blocks]);
desired_signal_power = helper_user_strength.*zz;
fade = complex(sqrt(1/2)*randn(num_helpers,num_users,num_blocks),...
    sqrt(1/2)*randn(num_helpers,num_users,num_blocks));
desired_signal_power_iid = repmat(desired_signal_power,[1,1,num_blocks]).*(abs(fade).^2);
helper_user_strength_iid = helper_user_strength_repeat.*(abs(fade).^2);
%***********
% Now, we consider the issue of frequency reuse. For any given frequency
% plan, we shouls single out the interference caused by a particular
% frequency. Though we assume that a user connects to a helper only if it
% is within 100m, it can see interference from a helper which is more than
% 100m away.
%**********************************


signal_plus_int_power_iid = repmat(sum(helper_user_strength_iid,1),[num_helpers,1,1]);
%******* THE BELOW CODE IS TO INTRODUCE FREQ REUSE*************
%temp = ones(num_cells*helpers_per_cell,num_cells*users_per_cell);
%temp(1,:) = 0;
%temp = cumsum(temp);
% signal_plus_int_power = zeros(num_helpers,num_users,freq_reuse);
% tempor = mod(temp,freq_reuse);
% for i = 1:freq_reuse
%     freq_id = (tempor == i-1);
%     signal_plus_int_power(:,:,i) =freq_id.*repmat(sum(helper_user_strength.*freq_id),num_helpers,1);
% 
% end
% 
% signal_plus_interf_power = sum(signal_plus_int_power,3);
%******************ABOVE CODE FOR FREQ REUSE**********************************
interf_power_iid = signal_plus_int_power_iid - desired_signal_power_iid;
SINR_iid = desired_signal_power_iid./(interf_power_iid+noise_power);
cap_mat = (total_symbols)*mean(log2(1+SINR_iid),3);
cap_bs = (total_symbols/100)*mean(log2(1+bs_user_strength_iid),1);

%cap_mat = (total_symbols)*mean(log2(1+SINR_iid),3);
%cap_bs = (total_symbols)*mean(log2(1+bs_user_strength_iid),1);
cap_bs(5:50) = 0;%the BS offers no rate to users 5:50

%****** the following lines of code is to pick only the best two helpers for
%every user
% cap_helpcumbs = [cap_mat;cap_bs]
% best = max(cap_helpcumbs);
% best = repmat(best,num_helpers+1,1);
% list_lebest1 = (cap_helpcumbs < best);
% list_lebest = cap_helpcumbs.*list_lebest1;
% second_best = max(list_lebest);
% second_best = repmat(second_best,num_helpers+1,1);
% best_index = (cap_helpcumbs == best)
% second_best_index = (cap_helpcumbs == second_best)
% zzz = [zz;ones(1,num_users)];
% num_nbhs = repmat(sum(zzz),num_helpers+1,1);
% more_than2 = (num_nbhs>2);
% best_index = best_index & more_than2;
% second_best_index = second_best_index & more_than2;
% tempora = cap_helpcumbs.*best_index;
% cap_helpcumbs(more_than2) = tempora(more_than2); 
% tempora = cap_helpcumbs.*second_best_index;
% cap_helpcumbs(more_than2) = tempora(more_than2);
% active_links = (cap_helpcumbs>0);
% cap_mat = cap_helpcumbs(1:num_helpers,:);
% cap_bs = cap_helpcumbs(num_helpers+1,:);
% zz = active_links(1:num_helpers,:);
active_links = 0;