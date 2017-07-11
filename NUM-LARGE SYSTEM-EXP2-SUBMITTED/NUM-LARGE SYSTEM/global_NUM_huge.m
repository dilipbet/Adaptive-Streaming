
clear all
% thus, we have 15 queues per cluster. The rates achievable in this cluster
% depends on the interference seen from surrounding clusters. Distortion
% corresponds to each user. Thus, there will be (5*number of clusters)
% users in the whole area. We can assume that the 5 users are uniformly
% distributed in the area of that cluster or almost equidistant from all
% the helpers in the cluster. each helper is allocated different frequency
% and it time shares among the 5 users. thus, there are 3 frequencies.
% further, each receiver sees interference on all the 3 frequencies from 2
% or 3 neighboring base stations.
%%  *********some system parameters**********
no_helper = 0; % NO HELPER IS 1 if there is only BS and no helpers and 0 if there are helpers
macro_side = 400;                           % side length of macro square grid
cell_side = 80;                             % side length of cell square
num_cells = (macro_side/cell_side)^2;       %total number of cells
helpers_per_cell = 1;
users_per_cell = 2;
W_lte = 20*(10^6);
W_wifi = 20*(10^6);

N = 1000;                                    %number of scheduling slots for which we 
                                            %run the simulation = number of
                                            %slots in one sample path
                                            
N1 = 1;                                     % number of randomized placements over 
                                            %which whole simulation is
                                            %averaged = number of sample
                                            %paths

%epsilon = 0.1;
freq_reuse = 1;
%param = 1^10;
param =  10^12*[ 10];
param_drop = [20 50 100 150];
param_del = [5 10 15 25];
num_helpers = num_cells*helpers_per_cell;
num_users = num_cells*users_per_cell;

trans_time_list = zeros(length(param),num_users);
worst_delay_list = zeros(length(param),num_users);
alluser_delay_profile = zeros(N,num_users,length(param));
helper_assign_eachv = zeros(N,num_users,length(param));
reception_profile = zeros(N,num_users,length(param));
bs_queues = zeros(N+1,num_users,length(param));
timeaveutil = zeros(1,length(param));
timeavetotbacklog = zeros(1,length(param));


%% ************ IMPORTING VIDEO DATA***********************************
% v1, v2, v3, v4 are 4 different (uncorrelated) videos and these are
% requested by 4 different users as placed in topology.mat


total_gop = 200;
v1 = importdata('video1_ssim.csv');
v2 = importdata('video2_ssim.csv');
v3 = importdata('video3_ssim.csv');
v4 = importdata('video4_ssim.csv');
v22 = zeros(total_gop*4,4);
v33 = zeros(total_gop*4,4);
%Since, additional bits of each layer are given in the imported files, to find the total bits
%corresponding to a particular quality, we need to take the cumulative sum.
%That is what we do in the following 'for loop'.
for dd = 1:total_gop
    v1((dd-1)*8+1:(dd-1)*8+8,4) = cumsum(v1((dd-1)*8+1:(dd-1)*8+8,4),1);
    v4((dd-1)*8+1:(dd-1)*8+8,4) = cumsum(v4((dd-1)*8+1:(dd-1)*8+8,4),1);
    v3((dd-1)*4+1:(dd-1)*4+4,4) = cumsum(v3((dd-1)*4+1:(dd-1)*4+4,4),1);
    v2((dd-1)*4+1:(dd-1)*4+4,4) = cumsum(v2((dd-1)*4+1:(dd-1)*4+4,4),1);
end
%v2 and v3 have only 4 layers. To make it compatible with the future code
%in congestion_control.m, we assume all videos have the same fixed number
%of layers = 8. Since videos 2 and 3 have 4 layers, we artificially append
%the 5th 6th 7th and 8th layers which all have the same quality as the 4th
%layer. Thus, the table for v2 and v3 is modified to v22 and v33 in the
%folloing for loop. Now, the new tables v22 and v33 have for each GOP, 4
%more artificial layers of best quality. Last 5 entries of the table will
%be same and will correspond to the best quality.
for tt = 1:total_gop
v22((tt-1)*8+1:(tt-1)*8+4,:) = v2((tt-1)*4+1:(tt-1)*4+4,:);
v22((tt-1)*8+5:(tt-1)*8+8,:) = repmat(v2(tt*4,:),4,1);
v33((tt-1)*8+1:(tt-1)*8+4,:) = v3((tt-1)*4+1:(tt-1)*4+4,:);
v33((tt-1)*8+5:(tt-1)*8+8,:) = repmat(v3(tt*4,:),4,1);
end
%v22 = circshift(v4,100*8+1);
%v33 = circshift(v1,100*8+1);
num_layers = 8*ones(1,num_users); % we make a list of the number of layers in each video
num_layers(1:4) = [8 4 4 8]; 
xx = [v1;v22;v33;v4];% we concatenate the tables of all video trace files to a single table
rand_start = linspace(16,800,num_users);% This indicates the starting point of
                                                     % video download for
                                                     % every user.
%rand_start(1:4) = [1 201 401 601]; %since we have 4 videos concatenated one after the other
                                   %and there are 4 users each requesting a
                                   %full video with 200 GOP's, we set the
                                   %start points to 1, 201, 401 and 601
                                   %resply
chunk_duration = 2/3;              % each GOP/chunk has duration 2/3 seconds;
pix_per_frame = 720*480; %pixels per frame vary with different videos
frame_per_sec = 24;                %frame rate is 24 frames per second and since each
                                   %GOP is of duration 2/3 seconds, we have
                                   %16 frames in each GOP.
slot_duration = chunk_duration; %setting scheduling slot duration equal to chunk duration
                                                                         
                                    
%****IMPORTING VIDEO DATA ENDS*************************



                                            

%rand_start(1:4) = [1 1 1 1];
%rand_start = ones(1,num_users);
%% *********** INITIALIZING PERFORMANCE METRICS**********************
util_ave = zeros(length(param),N);  %sum utility over users for every slot in every sample path
user_wise_util = zeros(N1,num_users);%time average utility for every sample path for
                                     %every user 
actual_util_time_ave = zeros(length(param),1);
delay_bound = zeros(1,4);
final_time_ave_q = zeros(num_helpers,num_users,N1);
total_backlog = zeros(N1,1,N);%sum backlog over users for every slot in every sample path
num_interrupt = zeros(N1,num_users);
trans_time = zeros(N1,num_users);
plback_avg_delay = zeros(N1,num_users);
frac_time_starved = zeros(N1,num_users);
%% LOADING TOPOLOGY, 
%origin is the centre of square grid and z_h,
% z_u indicate the distances of the helpers and the users from the origin
load('topology.mat','z_h','z_u');
%[z_h, z_u] = spatial_model(macro_side,cell_side,helpers_per_cell,users_per_cell);
%save('topology.mat','z_h','z_u');
%% *****SETTING UP PHYSICAL LAYER*****
[cap_mat,cap_bs,zz,active_links,zz_bs] = physical_layer(z_h,z_u,macro_side,cell_side,helpers_per_cell,users_per_cell...
    ,W_lte,W_wifi,freq_reuse,slot_duration,no_helper);
tt = zz;
%load('videodata.mat','xx');
%xx(225*8+1:225*8+24,:)=0;
%xx(all(xx==0,2),:)=[];
%size(xx)
%PSNR = xx(:,3);
%MSE = (255^2)./(10.^(PSNR/10));
%xx(:,3) = MSE;
active_users = ones(1,num_users);
jj=1;
%V1 = param;
del_param = 25;
drop_param = 50;
%% ***SWEEPING ACROSS NEELY PARAMETER V******
for V1= param   
for j = 1:N1
    

    
%if(no_helper)
 %   W_lte = 80*(10^6);
%end
%**************INITIALIZING QUEUES*********************************
z = zeros(num_helpers,num_users); %helper to user queues or lagrange multipliers
virt_q = zeros(num_helpers,num_users);%virtual queues from helpers to users
bs_queue = zeros(1,num_users); %BS to user queues 
theta = (V1/2)*ones(1,num_users);
virt_bs = zeros(1,num_users);% virtual queues from BS to users

%*************DROPPING PARAMETERS**************************
V = V1; %Neely parameter
beta = 2; %coefficient of the dropping objective
%R_max = max(xx(:,4));% The max possible arrival that can happen into a queue
%R_max = 375000*(2/3);
epsilon1 = (10^5)*ones(num_helpers,num_users); %the epsilon which is added to the virtual
                                               %queue as the actual queue
                                               %remains unserved
epsilon_bs = (10^5)*ones(1,num_users);         %the epsilon which is added to the virtual 
                                               %queues at the BS as the
                                               %actual queues at the BS
                                               %remain unserved
%Note that there is an O(V), O(1/V) worst case delay and utility tradeoff.
%The worst case delay is O(V/epsilon). However, this holds under the
%condition that epsilon is chosen to be less than R_max. Larger the
%epsilon, better the worst case delay, but larger will be the dropping
%rate.
%*************



%***SETTING UP THE SPATIAL PLACEMENT MODEL*******
%[z_h,z_u] = spatial_model(macro_side,cell_side,helpers_per_cell,users_per_cell);

%tempal = zeros(1,total_gop);
%for gh = 0:total_gop-1
%    tempal(gh+1) = sum(xx(8*(gh)+1:8*gh+8,4));
%end



%R_max = pix_per_frame*frame_per_sec*chunk_duration*0.5*log2(1/0.01);

bringhelper2 = zz(2,:);
%cap_mat = repmat(p2p_capacity,size(z));
mu = zeros(num_helpers,num_users); %SERVICE MATRIX OF QUEUES FROM HELPERS TO USERS
gamma = zeros(1,num_users);
mu_bs = zeros(1,num_users); %SERVICE VECTOR FROM BS TO USERS
dropped = zeros(num_helpers,num_users);%the matrix of d_{hu}
dropped_bs = zeros(1,num_users);% the vector of d_{hu} where h is the BS.
rr = zeros(num_helpers+1,num_users,N);% the request/arrival profile of all queues over all slots in a sample path
mumu = zeros(num_helpers+1,num_users,N);% the service profile of all queues over all slots in a sample path
trans = zeros(num_helpers+1,num_users,N);
offered = zeros(num_helpers+1,num_users,N);
droplist = zeros(num_helpers+1,num_users,N);% the drop profile of all queues over all slots in a sample path
util = zeros(1,N);
util2 = zeros(1,N);
actual_util = zeros(1,N);
totalbacklog = zeros(1,N);
q_evolve = zeros(num_helpers,num_users,N);
bs_evolve = zeros(N,num_users);
d_evolve = zeros(1,num_users,N);
gamma_evolve = zeros(1,num_users,N);
qual_evolve = zeros(N,num_users);
layer_evolve = zeros(N,num_users);
virt_evolve = zeros(num_helpers+1,num_users,N);
theta_evolve = zeros(N,num_users);
dwnld_over = ones(1,num_users);
req_time_stamp_list = zeros(N,num_users);
chunk_number_list = zeros(N,num_users);
chunk_number_requested = zeros(1,num_users);
    %****** QUEUE UPDATE****
    %[r,d_choice] = congestion_control(z,V,zz);
    %z = z + r;
 qual_choice_prev = num_layers; %intializing the choice of quality to best quality and will
 %indicate the choice in the previos slot asn time progresses. The
 %knowledge of previous quality choice is important because we shift the choice of quality by
 %only +1 or -1 to avoid the pumping effect
 %********BEGINNING THE DYNAMIC SIMULATION FOR 200 TIME SLOTS...
 t1 = 30;
 t2 = 50;
 t3 = 100;
 %even_vec = cumsum(2*ones(1,num_users/2));
 %active_users = ones(1,num_users);
 %req_time_stamp = zeros(1,num_users);
 %active_users(even_vec)=0;
 %zz = tt.*repmat(active_users,num_helpers,1);
 %request_list = zeros(N,num_users);

 %zz(:,1) = 0;
 request_list = zeros(N,num_users);
 req_time_stamp = zeros(1,num_users);
 path_x = zeros(1,N);
 path_y = zeros(1,N);
 %% ****BEGINNING DYNAMIC SIMULATION FOR N SLOTS******
for i = 1:N
    %if(i>N/2)
      %  zz = tt;
    %end
%     if(i==t1)
%         prev_bscap = cap_bs(1);
%         cap_bs(1) = cap_bs(1)/4;
%    
%      end
%     
    
%    if(i>t2)
%       no_helper = 0; 
%    end


    
%     if(i>t3 && i<=200)
%         no_helper = 0;
%        zz_bs(3) = 0;
%        zz(:,3) = 0;
%     end
    %if(no_helper)
     %   zz = 0;
    %end
    
     %% ******MOBILITY**************
     [z_h,z_u] = mobility(z_h,z_u);
     [cap_mat,cap_bs,zz,active_links,zz_bs] = physical_layer(z_h,z_u,macro_side,cell_side,helpers_per_cell,users_per_cell...
     ,W_lte,W_wifi,freq_reuse,slot_duration,no_helper);
    
    %% ********implementing the admission control decision************
    [r,d_choice,r_bs,qual_choice,r_choice] = congestion_control(z,V,zz,bs_queue,no_helper,pix_per_frame,...
        frame_per_sec,chunk_duration,num_users,num_helpers,zz_bs,xx,i,rand_start,total_gop,...
        qual_choice_prev,num_layers,theta,active_users);
   
    
    %% ********implementing the transmission scheduling decision*********
    [mu,mu_bs] = scheduler(z,cap_mat,zz,bs_queue,cap_bs,no_helper,zz_bs,num_users);
    
    
    %% *****gamma decisions******
    [gamma] = gammadec(theta,zz_bs,V);
   %% ***TRANSMIT QUEUE DYNAMICS********
    transmitted = min(z,mu);
    actual_served = transmitted; 
    z = z - actual_served + r;
    
    transmitted_bs = min(bs_queue,mu_bs);
    actual_served_bs = transmitted_bs;
     bs_queue = bs_queue - actual_served_bs + r_bs;
     
     theta = max(theta+gamma-d_choice,0);
    %r
    %d_choice
    %mu
    
    %z
    %qq = sum(sum(z))
    q_evolve(:,:,i) = z;
    bs_evolve(i+1,:) = bs_queue;
    theta_evolve(i,:) = theta;
    d_evolve(:,:,i) = d_choice;
    gamma_evolve(:,:,i) = gamma;
    qual_evolve(i,:) = qual_choice;
    layer_evolve(i,:) = r_choice;
    qual_choice_prev = qual_choice;
    virt_evolve(:,:,i) = [virt_q; virt_bs];
    %util(i) = sum(d_choice,2);
    util (i) = sum(d_choice(1:4));
    path_x(i) = real(z_u(1));
    path_y(i) = imag(z_u(1));
    %util2(i) = sum(sum(z));
    util2(i) = sum(sum(z.*zz));
    
    rr(:,:,i) = [r;r_bs];
    trans(:,:,i) = [transmitted;transmitted_bs];%matrix of actual transmitted
                                                %bits from all queues over
                                                %time.. it is a 3
                                                %dimensional matrix where
                                                %the 3rd dimension is time
    
    mumu(:,:,i) = [actual_served;actual_served_bs];% 3 dim matrix of transmitted)
   % indicate_chkinqueue = repmat(sum(mumu(:,:,1:i),3),[1,1,i])<cumsum(rrr(:,:,1:i));%head of line
    % the set of chunks that haven't yet been thrown out   
    %head_of_line = (cumsum(indicate_chkinqueue,3)==1);
    %[row,col] = find(head_of_line==1);
    %quotient = floor(col./num_users);
    %slot_index = quotient+1;
   actual_util(i) = sum(d_choice,2);
   totalbacklog(i) = sum(sum([z;bs_queue]));
   i
   req_time_stamp((active_users==1)) = i;
   chunk_number_requested((active_users==1))=chunk_number_requested...
       ((active_users==1))+1;

   request_list(i,:) = req_time_stamp;
   chunk_number_list(i,:) = chunk_number_requested;
end

%% ****** POST PROCESSING TO CALCULATE CHUNK DELAYS**************


[slot_index,user_index,allchunk_reception_profile,helper_assign,rrr,served_bits]...
     = playback_buffers(rr,mumu,num_users,N,num_helpers,trans,zz_bs);

 alluser_chunk_delay = allchunk_reception_profile - repmat(cumsum(ones(N,1),1),1,num_users);


%%
% arrival_profile = allchunk_reception_profile;
%  arrival_profile(arrival_profile==inf) = 0;
%  arrival_profile(alluser_chunk_delay>100) = 0;
%  playable_time = zeros(N,num_users);
%  for i = 1:N
%  playable_time(i,:) = max(arrival_profile(1:i,:),[],1);
%  end
%  playable_time(allchunk_reception_profile==inf) = inf;
%  playable_time(alluser_chunk_delay > 100) = inf;
%  num_avail = zeros(N,num_users);
%  for i = 1:N
%      avail_for_play = (playable_time <= i);
%      num_avail(i,:) = sum(avail_for_play);   
%  end
%  pb_buffers = zeros(N,num_users);
%  delay_profile = alluser_chunk_delay;
% % %delay_profile(delay_profile==inf) = 0;
%  max_delay_window = inf*ones(N,num_users);
%  for i = 1:N
%      max_delay_window(i,:) = max(delay_profile.*((max(0,i-10) < allchunk_reception_profile) & (allchunk_reception_profile <= i)),[],1);
%  end
%  max_delay_window(max_delay_window==0)= inf;
%  %start = (num_avail(1,:) > 3*max_delay_window(1,:));
%  pb_buffers(1,:) = num_avail(1,:);
%  start = zeros(1,num_users);
%  %start_point = zeros(1,num_users);
%  %start_point(start) = 1;
%  for i= 1:N-1
%      temp1 = pb_buffers(i,:);
%      temp2 = num_avail(i+1,:);
%      temp22 = num_avail(i,:);
%      temp3 = max_delay_window(i,:);
%      play_start = ((temp1 >= 3*temp3) &(start == 0));
%      %file_number = floor(chunk_number_list(i,:)/500);
%      %last_chunk_in_session = 500*file_number;
%      %last_chunk_location = (chunk_number_list == repmat(last_chunk_in_session,N,1));
%      %last_chunk_loc = (cumsum(last_chunk_location)==1);
%      %last_chunk_playable = sum(playable_time.*last_chunk_loc,1);
%      %stop = (temp1 == 0) & (active_users_mat(i-1,:)==0) & (i > last_chunk_playable);
%      %new_session_start = (mod(chunk_number_list(i,:),1000)==1);
%      new_startup = (pb_buffers(i,:)==0);
%      start(new_startup) = 0;
%      start(play_start)=1;
%      %start(stop) = 0;
%      temp1((start==1)) = max(temp1((start==1))-1,0) + temp2((start==1))-temp22((start==1));
%      temp1((start==0)) = temp1((start==0)) + temp2((start==0))-temp22((start==0));
%      pb_buffers(i+1,:) = temp1;
%      %start_point(new_start) = i;
%  end

%% *******SIMULATING PLAYBACK BUFFER PROCESS***************
arrival_profile = allchunk_reception_profile;
 arrival_profile(arrival_profile==inf) = 0;
 %arrival_profile(alluser_chunk_delay>100) = 0;
 playable_time = zeros(N,num_users);
 for i = 1:N
    playable_time(i,:) = max(arrival_profile(1:i,:),[],1);    
 end
 playable_time(allchunk_reception_profile==inf) = inf;
 arrival_profile = allchunk_reception_profile;
 
 num_avail = zeros(N,num_users);
 renewed_num_avail = zeros(N,num_users);
 num_arrived = zeros(N,num_users);
 
 pb_buffers = zeros(N,num_users);
 delay_profile = alluser_chunk_delay;
% %delay_profile(delay_profile==inf) = 0;
 max_delay_window = inf*ones(N,num_users);
 %max_delay_window(max_delay_window==0)= inf;
 %pb_buffers(1,:) = num_avail(1,:);
 start = ones(1,num_users);
 
 play_start_time = zeros(N,num_users);
 buff_start_time = zeros(N,num_users);
 last_playable_chunk = zeros(N,num_users);
 drop_list = zeros(N,num_users);
 newly_avail_for_play = zeros(N,num_users);
 renewed_playable_time = playable_time;
 last_playable = zeros(1,num_users);
 count = zeros(1,num_users);
 new_chunks_avail = zeros(N,num_users);
 drop = zeros(1,num_users);
 first_avail = zeros(N,num_users);
 for i= 2:N
     temporaryy = chunk_number_list.*((arrival_profile<=i)...
         & (chunk_number_list > repmat(last_playable,N,1)));                %find the chunk numbers which are greater than current last playable
                                                                            %chunk and are available now, i.e., time ***i**** 
     
     num_chunk_avail = sum(temporaryy~=0);                                  %number of new chunks which have arrived before ***i***
     new_chunks_avail(i,:) = num_chunk_avail;
     temporaryy(temporaryy==0)=inf;
     first_available = min(temporaryy);                                     %the chunk among the newly available chunks which comes first in
                                                                            % order 
     first_available(first_available==inf)=0;
     first_avail(i,:) = first_available;
     can_add = (first_available == (last_playable+1));                      %condition indicates that the first chunk in order is right next to 
                                                                            %the last playable chunk 
     cannot_add = (first_available > (last_playable+1));                    % the first available chunk is not next to the last playable chunk
     
     cond0 = (first_available <= last_playable);                            %this can happen when no new chunk is available
     cond1 = (num_chunk_avail>drop_param);                                          %the number of new chunks available is greater than 20
     cond2 = ~cond1;
     cond11 = (cannot_add & cond1);                                          
     cond22 = (cannot_add & cond2);
     cond111 = cond11 & (first_available-last_playable == 2);               %the condition that the first avail new chunk is 2 steps away from
                                                                            %last playable chunk 
     cond112 = cond11 & (first_available-last_playable > 2);                % greater than 2 steps away from last playable chunk
      
     %count(can_add) = 0;
     %count(cond11) = 0;
     %count(cond22) = count(cond22) + 1;
     
     arrived_list = (chunk_number_list >= repmat(first_available,N,1)) & (arrival_profile<=i);
     consec = [(diff(arrived_list) == -1);zeros(1,num_users)];
     temporr = chunk_number_list.*consec;
     temporr(temporr==0)=inf;
     last_consec = min(temporr);
     last_consec(last_consec==inf)=0;
     
     last_playable(can_add) = last_consec(can_add);
     last_playable(cond22) = last_playable(cond22);
     last_playable(cond111) = last_consec(cond111);
     last_playable(cond112) = last_playable(cond112)+1;
     last_playable_chunk(i,:) = last_playable;
     
     newly_avail = max(last_playable-first_available+1,0);
     newly_avail(cond0) = 0;
     newly_avail_for_play(i,:) = newly_avail;
     temp1 = pb_buffers(i-1,:);
     temp2 = newly_avail;
     
     drop(cond11) = drop(cond11)+1; 
     temptemp = (chunk_number_list == repmat(last_playable_chunk(i-1,:)+1,N,1));
     tempor = cumsum(temptemp);
     temptemp(tempor>1) = 0;
     drop_cond = (temptemp) & (repmat(cond11,N,1));
     drop_list(drop_cond) = 1;
     
     max_delay_window(i,:) = max(delay_profile.*((max(0,i-10) < allchunk_reception_profile)...
         & (allchunk_reception_profile <= i)),[],1);
     max_delay_window(max_delay_window==0)=inf;
     temp3 = max_delay_window(i,:);
     new_startup = ((pb_buffers(i-1,:) == 0) & (start==1)& (mod(chunk_number_list(i,:),1000)~=0));
     play_start = ((temp1 >= del_param*temp3) &(start == 0));
     temp_start = play_start_time(i,:);
     temp_stop = buff_start_time(i,:);
     temp_start(play_start) = i;
     temp_stop(new_startup) = i;
     play_start_time(i,:) = temp_start;
     buff_start_time(i,:) = temp_stop;
     %play_time(new_start) = play_time(new_start)+1;
     %file_number = floor(chunk_number_list(i,:)/500);
     %last_chunk_in_session = 500*file_number;
     %last_chunk_location = (chunk_number_list == repmat(last_chunk_in_session,N,1));
     %last_chunk_loc = (cumsum(last_chunk_location)==1);
     %last_chunk_playable = sum(playable_time.*last_chunk_loc,1);
     %stop = (temp1 == 0) & (active_users_mat(i-1,:)==0) & (i > last_chunk_playable);
     %new_session_start = (mod(chunk_number_list(i,:),500)==1);
     
     %start(new_session_start) = 0;
     start(new_startup) = 0;
     start(play_start)=1;
     %start(stop) = 0;
     temp1((start==1)) = max(temp1((start==1))-1,0) + temp2((start==1));
     temp1((start==0)) = temp1((start==0)) + temp2((start==0));
     pb_buffers(i,:) = temp1;
     i
     %start_point(new_start) = i;
 end
%ecdf(sum(alluser_chunk_delay>100));

num_chunks_requested = chunk_number_list(N,:);
dropped_chunks = chunk_number_list.*drop_list;
frac_dropped = drop./num_chunks_requested;
num_not_dropped = num_chunks_requested-drop;
ssim_evolve = permute(d_evolve,[3 2 1]);
timeave_ssim = sum(ssim_evolve.*(drop_list==0))./num_not_dropped;
%%
total_rebuff_time = zeros(1,num_users);
startup_time = zeros(1,num_users);
num_rebuff = zeros(1,num_users);
ave_rebuff_time = zeros(1,num_users);
num_sessions = ceil(num_chunks_requested./1000);
cum_num_sessions = cumsum(num_sessions);
frac_rebuff_sess = zeros(1, sum(num_sessions));
num_rebuff_sess = zeros(1, sum(num_sessions));
startup_time_sess = zeros(1, sum(num_sessions));
frac_drop_sess = zeros(1,sum(num_sessions));
fracdrop = zeros(N/1000,num_users);
for hh = 1:N/1000
 fracdrop(hh,:) = sum(ceil(dropped_chunks/1000)==hh)/1000;   
end
for ppp = 1:num_users
    if(ppp ==1) 
    frac_drop_sess(1:num_sessions(1)) = fracdrop(1:num_sessions(1),1)';
    else
        frac_drop_sess(cum_num_sessions(ppp-1)+1:cum_num_sessions(ppp)) = fracdrop(1:num_sessions(ppp),ppp)';
    end
end
for i = 1:num_users
    xuxu = buff_start_time(:,i);
    xxx = chunk_number_list(:,i).*(xuxu>0);
    prefetch_start = xuxu(mod(xxx,1000)==1);
    buff_start_list = xuxu(xuxu~=0);
    yy = play_start_time(:,i);
    play_start_list = yy(yy~=0);
    if(length(buff_start_list) - length(play_start_list) > 0)
        play_start_list = [play_start_list;1000];
    end
    buffering_time = play_start_list-buff_start_list;
    
    
    startup_time(i) = buffering_time(1);
    total_rebuff_time(i) = sum(buffering_time)-startup_time(i);
    num_rebuff(i) = length(buffering_time)-1;
    ave_rebuff_time(i) = mean(buffering_time(2:length(buffering_time)));
     
end
  total_play_time = 1000;
  frac_rebuff_time = total_rebuff_time./total_play_time;
  ave_rebuff_time(total_rebuff_time==0)=0;



end

figure(1)
ecdf(frac_dropped*100)
hold all

figure(2)
ecdf(startup_time)
hold all

figure(3)
ecdf(num_rebuff)
hold all

figure(4)
ecdf(frac_rebuff_time*100)
hold all

figure(5)
ecdf(timeave_ssim)







jj = jj+1;
hold all
end
jorara = permute(d_evolve, [3 2 1]);
jorara2 = permute(sum(rr,1), [3 2 1]);


% subplot(2,1,1);
% plot((2*param*beta+R_max+10^5)/(10^5))
% subplot(2,1,2);
% plot(actual_util_time_ave);
