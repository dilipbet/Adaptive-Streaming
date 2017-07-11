
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
%param = 10^10;
param = linspace(10^11, 10^12, 5);
%param  = 10^10;
 
alluser_delay_profile = zeros(N,4,length(param));
helper_assign_eachv = zeros(N,4,length(param));
reception_profile = zeros(N,4,length(param));
bs_queues = zeros(N+1,4,length(param));
timeaveutil = zeros(1,length(param));
timeavemodutil = zeros(1,length(param));
timeavetotbacklog = zeros(1,length(param));
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
%*****SETTING UP PHYSICAL LAYER*****
[cap_mat,cap_bs,zz,active_links,zz_bs] = physical_layer(z_h,z_u,macro_side,cell_side,helpers_per_cell,users_per_cell...
    ,W_lte,W_wifi,freq_reuse,slot_duration,no_helper);
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
theta = (V1/2)*ones(1,num_users);
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
sum_qual = zeros(1,N);
totalbacklog = zeros(1,N);
sum_util = zeros(1,N);
q_evolve = zeros(num_helpers,num_users,N);
bs_evolve = zeros(N,num_users);
d_evolve = zeros(1,num_users,N);
gamma_evolve = zeros(1,num_users,N);
qual_evolve = zeros(N,num_users);
layer_evolve = zeros(N,num_users);
virt_evolve = zeros(num_helpers+1,num_users,N);
theta_evolve = zeros(N,num_users);
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
        qual_choice_prev,num_layers,theta);
   
    
    %********implementing the transmission scheduling decision*********
    [mu,mu_bs] = scheduler(z,cap_mat,zz,bs_queue,cap_bs,no_helper,zz_bs);
    
    
    %*****gamma decisions******
    [gamma] = gammadec(theta,zz_bs,V);
   
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
   %sum_qual(i) = sum(d_choice,2);
   totalbacklog(i) = sum(bs_queue(1:4),2);
   sum_util(i) = sum(log(1+gamma(1:4)));
end



temporary = cumsum(d_evolve,3)./cumsum(ones(1,num_users,N),3);
user_wise_util(j,:) = temporary(:,:,N);
time_ave_q = cumsum(q_evolve,3)./cumsum(ones(num_helpers,num_users,N),3);
final_time_ave_q(:,:,j) = time_ave_q(:,:,N);
total_backlog(j,1,:) = sum(sum(time_ave_q,1),2);
[slot_index,user_index,allchunk_reception_profile,helper_assign,rrr,served_bits]...
     = playback_buffers(rr,mumu,num_users,N,num_helpers,trans,zz_bs);






end

util_ave_over_placement = sum(util_ave,1)/N1;
%user_wise_util_sorted = sort(user_wise_util,2);
user_wise_util_placement = sum(user_wise_util,1)/N1;
total_backlog_ave_placement = sum(total_backlog,1)/N1;

aa = sort(user_wise_util_placement);
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
% video_qual_mat = zeros([N,4,8]);
% for iter = 1:8
%     video_qual_mat(:,:,iter) = [v1(8*ttt+iter,4) v22(8*ttt+iter,4) v33(8*ttt+iter,4) v4(8*ttt+iter,4)];
% end
% video_qual_mat(:,2:3,5:8)=inf;

%actual_qevolve = sum(repmat(received_chunkbits(:,1:4),[1,1,8])>=video_qual_mat,3);

%util_ave(jj,:) = cumsum(util)./cumsum(ones(1,N));
%actual_util_time_ave(jj) = sum(actual_util)/N;
%time_ave_qual = sum(d_evolve,3)/N;
%total_util = sum(log(1+time_ave_qual),2);
alluser_chunk_delay = allchunk_reception_profile(:,1:4) - repmat(cumsum(ones(N,1),1),1,4);
alluser_delay_profile(:,:,jj) = alluser_chunk_delay;
reception_profile(:,:,jj) = allchunk_reception_profile(:,1:4);
helper_assign_eachv(:,:,jj) = helper_assign(:,1:4);
bs_queues(:,:,jj) = bs_evolve(:,1:4);
timeaveutil(jj) = sum(log(1+mean(d_evolve,3)));
timeavemodutil(jj) = mean(sum_util); %utility of modified problem
timeavetotbacklog(jj) = mean(totalbacklog);
jj = jj+1;
%plot(cumsum(actual_util)./cumsum(ones(1,N)));
%plot(cumsum(totalbacklog)./cumsum(ones(1,N)));
plot(totalbacklog)
hold all
end
% subplot(2,1,1);
% plot((2*param*beta+R_max+10^5)/(10^5))
% subplot(2,1,2);
% plot(actual_util_time_ave);
