
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
%*********some system parameters**********
no_helper = 1; % NO HELPER IS 1 if there is only BS and no helpers and 0 if there are helpers
macro_side = 400;                           % side length of macro square grid
cell_side = 80;                             % side length of cell square
num_cells = (macro_side/cell_side)^2;       %total number of cells
helpers_per_cell = 1;
users_per_cell = 2;
W_lte = 20*(10^6);
W_wifi = 20*(10^6);

N = 200;                                    %number of scheduling slots for which we 
                                            %run the simulation = number of
                                            %slots in one sample path
                                            
N1 = 1;                                     % number of randomized placements over 
                                            %which whole simulation is
                                            %averaged = number of sample
                                            %paths

%epsilon = 0.1;
freq_reuse = 1;
param = 10^8;
 
alluser_delay_profile = zeros(N,4,length(param));
num_helpers = num_cells*helpers_per_cell;
num_users = num_cells*users_per_cell;

%%************ IMPORTING VIDEO DATA***********************************
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
rand_start = floor(unifrnd(1,total_gop,1,num_users));% This indicates the starting point of
                                                     % video download for
                                                     % every user.
rand_start(1:4) = [1 201 401 601]; %since we have 4 videos concatenated one after the other
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
%*********** INITIALIZING PERFORMANCE METRICS**********************
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
% loading the topology, i.e., origin is the centre of square grid and z_h,
% z_u indicate the distances of the helpers and the users from the origin
load('topology.mat','z_h','z_u');
%load('videodata.mat','xx');
%xx(225*8+1:225*8+24,:)=0;
%xx(all(xx==0,2),:)=[];
%size(xx)
%PSNR = xx(:,3);
%MSE = (255^2)./(10.^(PSNR/10));
%xx(:,3) = MSE;
jj=1;

for V1 = param   
for j = 1:N1
    

    
%if(no_helper)
 %   W_lte = 80*(10^6);
%end
%**************INITIALIZING QUEUES*********************************
z = zeros(num_helpers,num_users); %helper to user queues or lagrange multipliers
virt_q = zeros(num_helpers,num_users);%virtual queues from helpers to users
bs_queue = zeros(1,num_users); %BS to user queues 
virt_bs = zeros(1,num_users);% virtual queues from BS to users

%*************DROPPING PARAMETERS**************************
V = V1; %Neely parameter
beta = 2; %coefficient of the dropping objective
R_max = max(xx(:,4));% The max possible arrival that can happen into a queue
R_max = 375000*(2/3);
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
%*****SETTING UP PHYSICAL LAYER*****
[cap_mat,cap_bs,zz,active_links,zz_bs] = physical_layer(z_h,z_u,macro_side,cell_side,helpers_per_cell,users_per_cell...
    ,W_lte,W_wifi,freq_reuse,slot_duration);
bringhelper2 = zz(2,:);
%cap_mat = repmat(p2p_capacity,size(z));
mu = zeros(num_helpers,num_users); %SERVICE MATRIX OF QUEUES FROM HELPERS TO USERS
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

q_evolve = zeros(num_helpers,num_users,N);
bs_evolve = zeros(N,num_users);
d_evolve = zeros(1,num_users,N);
qual_evolve = zeros(N,num_users);
layer_evolve = zeros(N,num_users);
virt_evolve = zeros(num_helpers+1,num_users,N);
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
for i = 1:N
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
    %********implementing the admission control decision************
    [r,d_choice,r_bs,qual_choice,r_choice] = congestion_control(z,V,zz,bs_queue,no_helper,pix_per_frame,...
        frame_per_sec,chunk_duration,num_users,num_helpers,zz_bs,xx,i,rand_start,total_gop,...
        qual_choice_prev,num_layers);
   
    
    %********implementing the transmission scheduling decision*********
    [mu,mu_bs] = scheduler(z,cap_mat,zz,bs_queue,cap_bs,no_helper,virt_q,virt_bs,zz_bs);
    
    %********implementing the drop decision********************
    [dropped, dropped_bs] = drop(z,virt_q,V,beta,bs_queue,virt_bs,R_max,num_helpers,num_users,zz,zz_bs);
    
    %drop_ind = (dropped > 0); % the set of queues for which the algo chooses to drop in the 
    % current slot 
    transmitted = min(z,mu);
    dropped1 = min(z-transmitted,dropped);
    actual_served = transmitted + dropped1;
    %non_empty = (z>0);
    %virt_q(non_empty) = max(virt_q(non_empty)-mu(non_empty)-dropped(non_empty)+...
    %   epsilon1(non_empty),0);
    %virt_q(~non_empty) = max(virt_q(~non_empty)-mu(~non_empty)-dropped(~non_empty),0);
    non_empty = (mu+dropped < z);
    virt_q(non_empty) = max(virt_q(non_empty)-mu(non_empty)-dropped(non_empty)+...
        epsilon1(non_empty),0);
    virt_q(~non_empty) = 0;
    z = z - actual_served + r;
    
    transmitted_bs = min(bs_queue,mu_bs);
    dropped_bs1 = min(bs_queue-transmitted_bs,dropped_bs);
    actual_served_bs = transmitted_bs+dropped_bs1;
    %non_empty_bs = (bs_queue>0);
    %virt_bs(non_empty_bs) = max(virt_bs(non_empty_bs)-mu_bs(non_empty_bs)-dropped_bs(non_empty_bs)+epsilon_bs(non_empty_bs),0);
    %virt_bs(~non_empty_bs) = max(virt_bs(~non_empty_bs)-mu_bs(~non_empty_bs)-dropped_bs(~non_empty_bs),0);
    %bs_queue = bs_queue - actual_served_bs + r_bs;
     non_empty_bs = (mu_bs+dropped_bs<bs_queue);
     virt_bs(non_empty_bs) = max(virt_bs(non_empty_bs)-mu_bs(non_empty_bs)-dropped_bs(non_empty_bs)+epsilon_bs(non_empty_bs),0);
     virt_bs(~non_empty_bs) = 0;
     bs_queue = bs_queue - actual_served_bs + r_bs;
    %r
    %d_choice
    %mu
    
    %z
    %qq = sum(sum(z))
    q_evolve(:,:,i) = z;
    bs_evolve(i+1,:) = bs_queue;
    d_evolve(:,:,i) = d_choice;
    qual_evolve(i,:) = qual_choice;
    layer_evolve(i,:) = r_choice;
    qual_choice_prev = qual_choice;
    virt_evolve(:,:,i) = [virt_q; virt_bs];
    %util(i) = sum(d_choice,2);
    util (i) = sum(d_choice(1:4));
    %util2(i) = sum(sum(z));
    util2(i) = sum(sum(z.*zz));
    
    rr(:,:,i) = [r;r_bs];
    trans(:,:,i) = [transmitted;transmitted_bs];%matrix of actual transmitted
                                                %bits from all queues over
                                                %time.. it is a 3
                                                %dimensional matrix where
                                                %the 3rd dimension is time
    droplist(:,:,i) = [dropped1;dropped_bs1];     %matrix of actual dropped bits
                                                %from all queues over time
    mumu(:,:,i) = [actual_served;actual_served_bs];% 3 dim matrix of (dropped
     offered(:,:,i) = [mu;mu_bs];                                              %+transmitted)
   % indicate_chkinqueue = repmat(sum(mumu(:,:,1:i),3),[1,1,i])<cumsum(rrr(:,:,1:i));%head of line
    % the set of chunks that haven't yet been thrown out   
    %head_of_line = (cumsum(indicate_chkinqueue,3)==1);
    %[row,col] = find(head_of_line==1);
    %quotient = floor(col./num_users);
    %slot_index = quotient+1;
   actual_util(i) = sum(d_choice,2)+beta*sum(sum(droplist(:,1:4,i))); 
end



temporary = cumsum(d_evolve,3)./cumsum(ones(1,num_users,N),3);
user_wise_util(j,:) = temporary(:,:,N);
time_ave_q = cumsum(q_evolve,3)./cumsum(ones(num_helpers,num_users,N),3);
final_time_ave_q(:,:,j) = time_ave_q(:,:,N);
total_backlog(j,1,:) = sum(sum(time_ave_q,1),2);
[slot_index,user_index,allchunk_reception_profile,helper_assign,rrr,served_bits,...
    received_chunkbits,requested_chunkbits,dropped_chunkbits] = playback_buffers(rr,mumu,num_users,N,num_helpers,trans,droplist,zz_bs);


%plot(util_ave_over_placement)


%*****PLAYBACK DYNAMICS*****************
pb_buffers = zeros(1,num_users);
pb_rate = frame_per_sec*slot_duration*ones(1,num_users);
wait_count = zeros(1,num_users);
dwnld_position = zeros(1,num_users);
earliest_ind_rep = zeros(N,num_users);
earliest_availchunk_ind = zeros(1,num_users);
consec_dwnld_end = zeros(1,num_users);
pb_buffer_profile = zeros(N,num_users);
jaffa = cumsum(ones(N,num_users));
for i = 1:N
    %****PLAYBACK UPDATE****
    pb_buffers = max(pb_buffers - pb_rate,0); 
    
    %******ZEROING IN ON THE SEQUENCE OF CHUNKS WHICH CAN BE DOWNLOADED
    %WITHOUT GAPS******
    available_for_play = (allchunk_reception_profile <= i);%find all chunks available for
    %play at time slot 'i'
    available_for_play = available_for_play.*(received_chunkbits>0);
    boffa = (jaffa >repmat(dwnld_position,N,1));% all the indices greater than the current 
   %download position
   available_for_play = available_for_play.*boffa;% the chunks beyond current download
   %position which are available for play
   temporary = sum(available_for_play);%number of chunks beyond dwnld position avail for play
   num_chunks_avail = repmat(temporary,N,1);
   earliest_availchunk_ind(temporary==0) = 0;%assign zero to those users which have no chunk 
   %available beyond dwnld position
   temporary2 = available_for_play.*(num_chunks_avail > 0);% the list of chunks avail for play
   %for users which have something avail beyond dwnld position
   %earliest_ind_rep(num_chunks_avail==0) = 0;
   %[chunk_ind col1] = find(available_for_play.*(num_chunks_avail == 1));
   %earliest_availchunk_ind(temporary==1) = chunk_ind';
   %earliest_ind_rep(num_chunks_avail==1) = repmat(chunk_ind,N,1);
   %temporary2 = available_for_play.*(num_chunks_avail > 1);
   tempotempo = cumsum(temporary2);%note that there could exist a gap between 1st next and
   %2nd next chunk and therefore when we do cumsum, we could see a list of
   %1's before the 2nd available chunk
   ind500 = (tempotempo == 1);%finding the index of the first avail chunk beyond dwnld 
   %position
   ind501 = (sum(ind500)==1);%chunk1 and chunk2 are available successively
   ind502 = (sum(ind500) > 1);%there is a gap between availability of chunk1 and chunk 2
   [row1 col1] = find(ind500.*repmat(ind501,N,1)==1);
   earliest_availchunk_ind(ind501) = row1';% find the chunk number which is available at the
   %earliest for those users who don't have a gap in chunk number between
   %next and 2nd next.
   temporar = cumsum(ind500.*repmat(ind502,N,1));
   [row2 col2] = find(temporar == 1);
   earliest_availchunk_ind(ind502) = row2'; %find the chunk number available at earliest
   %for those users who have a gap between available successive chunks
   %earliest_ind_rep(num_chunks_avail>1) = repmat(row2,N,1);
    %[row col] = find(cumsum(available_for_play)==1);
    %earliest_availchunk_ind = earliest_ind_rep(1,:);
    temporary3 = available_for_play.*(num_chunks_avail > 0);
    earliest_ind_rep = repmat(earliest_availchunk_ind,N,1);
    temptemp = cumsum(temporary3 == 0);%find all chunk indices not available for play at 
    %slot i.
    consec_dwnld_point = (temptemp == earliest_ind_rep.*(num_chunks_avail>0));
    consec_dwnld_point = consec_dwnld_point.*(num_chunks_avail>0);
    ind40 = (sum(consec_dwnld_point)==1);
    ind41 = repmat(ind40,N,1);
    ind42 = (sum(consec_dwnld_point) >1);
    ind43 = repmat(ind42,N,1);
    [row3 col3] = find(consec_dwnld_point.*(ind41)==1);% find the index which is immediately
    %after the set of chunks which is successively downloadable from
    %earliest available chunk
    [row4 col4] = find(cumsum(consec_dwnld_point.*(ind43))==1);
    consec_dwnld_end(temporary==0) = earliest_ind_rep(temporary==0);
    consec_dwnld_end(ind40) = row3';
    consec_dwnld_end(ind42) = row4';
    
   
    ind1 = (pb_buffers ~= 0);
    ind2 = (earliest_availchunk_ind == dwnld_position+1);
    ind3 = ~ind1;
    ind4 = (wait_count < 5);
    ind5 = (wait_count == 5);
    ind8 = (wait_count == 5+1);
    
   wait_count(ind3 & ind8 & (~ind2)) = 0;
    %**** STILL WAITING FOR NEXT CHUNK IN SEQUENCE****
    ind6 = ind3 & ind4 & (~ind2);%If empty buffer&wait < max allowed wait&next chunk 
    %not available, then wait for one more slot
    wait_count(ind6) = wait_count(ind6) + 1;
   
    
    %****DECLARE CHUNK LOSS****
    ind7 = ind3 & ind5 & (~ind2);%empt buffer & wait = wait max & next chunk in sequence
    %not available, then declare chunk loss, update download position by +1
    % and change wait to 0
   
    wait_count(ind7) = 0;
    dwnld_position(ind7) = dwnld_position(ind7) + 1;
    
    %***NEXT CHUNK IN SEQUENCE IS AVAILABLE***
    dwnld_position(ind2) = dwnld_position(ind2) +...
        consec_dwnld_end(ind2) - earliest_availchunk_ind(ind2);% change dwnld position
    %last chunk which can be downloaded consecutively starting from
    %earliest available chunk
    pb_buffers(ind2) = pb_buffers(ind2) +...
        frame_per_sec*chunk_duration*(consec_dwnld_end(ind2) - earliest_availchunk_ind(ind2)); 
    wait_count(ind2) = 5+1;
    ind20 = ind1 & (~ind2);
    pb_buffers(ind20) = pb_buffers(ind20);
    wait_count(ind20) = 5+1;
    pb_buffer_profile(i,:) = pb_buffers;
end
digi_signal = (pb_buffer_profile > 0);% digital signal which indicates with '0' when pb buffer is 
                               % is empty and with '1' when pb buffer is
                               % non empty
signal_jump_locations = diff(digi_signal);
signal_jump2 = [signal_jump_locations;zeros(1,num_users)];
num_interruptions = sum(signal_jump_locations == -1);
%aa = sort(user_wise_util_placement);
%
%cc = sort(num_interruptions);
row_indices = cumsum(ones(N,num_users));
digisig2 = cumsum(digi_signal);
plback_delays_cumsum = (row_indices - digisig2).*(signal_jump2 == 1);
transient_time = row_indices.*(signal_jump2 == 1);
 
 
num_interrupt(j,:) = num_interruptions;
trans_time(j,:) = max(transient_time);
plback_avg_delay(j,:) = max(plback_delays_cumsum)./sum(plback_delays_cumsum >0);
frac_time_starved(j,:) = max(plback_delays_cumsum)/N;
end

util_ave_over_placement = sum(util_ave,1)/N1;
%user_wise_util_sorted = sort(user_wise_util,2);
user_wise_util_placement = sum(user_wise_util,1)/N1;
total_backlog_ave_placement = sum(total_backlog,1)/N1;
plback_ave_delay_plcmnt = sum(plback_avg_delay,1)/N1;
trans_time_plcmnt = sum(trans_time,1)/N1;
num_interrupt_plcmnt = sum(num_interrupt,1)/N1;
frac_time_starved_plcmnt = sum(frac_time_starved,1)/N1;
aa = sort(user_wise_util_placement);
cc = sort(plback_ave_delay_plcmnt);
dd = sort(trans_time_plcmnt);
ee = sort(num_interrupt_plcmnt);
ff = sort(frac_time_starved);
bb = cumsum(ones(1,num_users));
sth = permute(d_evolve,[3 2 1]);
%stairs(aa,bb);
%plot(cumsum(vec)./cumsum(ones(1,N)))
% grid on
% subplot(2,2,1),plot(qual_evolve(:,1));
% grid
% subplot(2,2,2),plot(qual_evolve(:,2));
% grid
% subplot(2,2,3),plot(qual_evolve(:,3));
% grid
% subplot(2,2,4),plot(qual_evolve(:,4));
% grid
%plot(qual_evolve(:,1));
%hold on
%comet(qual_evolve(:,1));
ttt = 0:N-1;
ttt = ttt';
video_qual_mat = zeros([N,4,8]);
for iter = 1:8
    video_qual_mat(:,:,iter) = [v1(8*ttt+iter,4) v22(8*ttt+iter,4) v33(8*ttt+iter,4) v4(8*ttt+iter,4)];
end
video_qual_mat(:,2:3,5:8)=inf;

actual_qevolve = sum(repmat(received_chunkbits(:,1:4),[1,1,8])>=video_qual_mat,3);

util_ave(jj,:) = cumsum(util)./cumsum(ones(1,N));
actual_util_time_ave(jj) = sum(actual_util)/N;
alluser_chunk_delay = allchunk_reception_profile(:,1:4) - repmat(cumsum(ones(N,1),1),1,4);
alluser_delay_profile(:,:,jj) = alluser_chunk_delay;
jj = jj+1;
plot(cumsum(actual_util)./cumsum(ones(1,N)));
hold all

end
% subplot(2,1,1);
% plot((2*param*beta+R_max+10^5)/(10^5))
% subplot(2,1,2);
% plot(actual_util_time_ave);
