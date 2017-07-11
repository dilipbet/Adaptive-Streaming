function[r,d_choice,r_bs,qual_choice,r_choice]=congestion_control(z,V,zz,bs_queue,no_helper,pix_per_frame,...
    frame_per_sec,chunk_duration,num_users,num_helpers,zz_bs,xx,i,rand_start,total_gop,...
    qual_choice_prev,num_layers,theta,active_users)
z(zz==0) = inf;% all the inactive edges in the bipartite graph are made infinite queue lengths
               % so that these queues (rather the corresponding helpers) are
               % never chosen by a user

temp = min(z,[],1); % recall z is the helper by user incidence matrix of queue lengths. min
                    %along the 1st dimension means choosing the minimum
                    %queue length of every column. This means that each
                    %user (each column) chooses the helper in its
                    %neighborhood (finite queue length) with the smallest
                    %backlog. In the current scenario, we have only 4 users
                    % 2 helpers and a base station. Thus, only the 1st 4
                    % columns and the 1st two rows are active. Thus, the
                    % 1st 4 columns will choose either helper 1 or helper 2
                    % depending on connectivity (i.e. whether the incidence
                    % matrix of the bipartite graph has a zero or 1 for the
                    % particular edge) and also depending on the queue
                    % length. The remainign dummy users and dummy helpers
                    % have queue lengths = inf. But, we should make sure
                    % they don't affect the performance of the firt 4 users
x = size(z);
best_helper_id = (z==repmat(temp,x(1),1)); % find the location of the best helper for 
                                           % each user. In case of a tie,
                                           % choose the first one which
                                           % comes first on the column list
                                           % now, here we can see the dummy
                                           % users will all choose the
                                           % first helper. We should make
                                           % sure the 1st helper while scheduling is not
                                           % swamped by this artificiality
                                           % (see scheduler.m)
%best_helper_id = best_helper_id.*zz;
tempor = cumsum(best_helper_id);
%tempora = cumsum(tempor);
best_helper_id(tempor>1) = 0;% this is to choose the 1st helper on the list in case of a tie
%num_best_helpers = sum(best_helper_id);
%best_qlength = sum(z.*best_helper_id)./num_best_helpers;
z(z==inf)=0;
best_qlength = sum(z.*best_helper_id);% find the smallest queue length in the neighborhood
                                      % for each user.


%if(i>50 && i<=100)
    %id(3:4) = logical(1);%helper 1 enters but users 3 and 4 still
     % download from base station
%end
%% *** IF WE  WANT TO INVOLVE BS*********
% bs_queue(zz_bs==0)=inf;% all the inactive edges in the BS to user graph are made infinite 
% id = best_qlength > bs_queue; % check whether the best among neighboring helpers is also
%                                % better than the BS. If so, request the
%                                % chunk from that helper. otherwise, request
%                                % it from the BS. 
% best_qlength(id) = bs_queue(id); % if BS is less congested, change the best queue length to
%                                  % the BS queue length
% idid = repmat(id,x(1),1);
% best_helper_id (idid) = 0;
% 
% 
%                                                                 
% 
% 
% 
% 
% if(no_helper)
%     best_qlength = bs_queue;
% end

%% ******* FOR VBR**********************************
dvec_t = zeros(1,8);
quality = cumsum(ones(8,1),1);
qual_rep = repmat(quality,1,x(2));
rand_start_rep = repmat(rand_start,8,1);
rand_start_rep = mod((rand_start_rep+i-1),total_gop*4);
rand_start_rep(rand_start_rep==0) = total_gop*4;
oppa = cumsum(repmat(ones(8,1),1,num_users),1);
lin_indices1 = 2*total_gop*4*8 + 8*(rand_start_rep-1)+oppa;
lin_indices2 = 3*total_gop*4*8 + 8*(rand_start_rep-1)+oppa;
dvec_t_rep = xx(lin_indices1);
chunk_size_rep = xx(lin_indices2);
%**********************************************

%chunk_size_rep = repmat(chunk_size,1,x(2));
%dvec_t_rep = repmat(dvec_t,1,x(2));
%% ***********FOR CBR: SOME PAPER BY WIEGAND**************************************************
%*************************************************************

% quality = cumsum(ones(3,1),1);% column vector starting from 1 and ending at 8 NOTE: 8 is changed to 3 for large scale CBR simulation
% qual_rep = repmat(quality,1,x(2));
% dvec_t = [20;27;36];
% dvec_t_rep = repmat(dvec_t,1,x(2));
% bitrates = 2000/3*[125;275;375];
% chunk_size_rep = repmat(bitrates,1,x(2));

%*****************************
%% ******* FOR CBR of FOREMAN SEQUENCE: SOME PAPER BY CISCO PEOPLE********
% quality = cumsum(ones(10000,1),1);
% qual_rep = repmat(quality,1,x(2));
% R_min = 1000*100;
% R_max = 1000*2154.3;
% bitrates = linspace(R_min,R_max,10000);
% bitrates = bitrates';
% chunk_size_rep = repmat(bitrates,1,x(2));
% alpha = 0.13;
% beta = 0.63;
% dvec_t = exp(-alpha.*(bitrates./R_max).^(-beta)+alpha);
% dvec_t_rep = repmat(dvec_t,1,x(2));
%*************
%% 
objec = repmat(best_qlength,length(dvec_t),1).*chunk_size_rep-repmat(theta,length(dvec_t),1).*dvec_t_rep;
%repmat(theta,length(dvec_t),1)
opt_id = (repmat(min(objec),length(dvec_t),1)==objec);
%% ****SPECIAL LINES FOR V2 AND V3****
temp_id = cumsum(opt_id);
opt_id(temp_id>1) = 0;
%%
qual_choice = sum(qual_rep.*opt_id);
d_choice = sum(dvec_t_rep.*opt_id);
r_choice = sum(chunk_size_rep.*opt_id);
d_choice = d_choice.*active_users;
qual_choice = qual_choice.*active_users;
r_choice = r_choice.*active_users;
%r_choice = pix_per_frame*frame_per_sec*chunk_duration*0.5*log2(1./d_choice);
r = best_helper_id.*repmat(r_choice,x(1),1);

r = r.*zz;


%% ****IF WE WANT TO INVOLVE BS*********
% r_bs = r_choice.*id;
% r_bs = r_bs.*zz_bs;
% if(no_helper)
%     r_bs = r_choice;
%     r_bs = r_bs.*zz_bs;
%     r = zeros(num_helpers,num_users);
%     r = r.*zz;
% end
r_bs = zeros(1,num_users);