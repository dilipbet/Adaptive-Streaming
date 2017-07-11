function[slot_index,user_index,allchunk_reception_profile,helper_assign,rrr,served_bits]...
     = playback_buffers(rr,mumu,num_users,N,num_helpers,trans,zz_bs)

%actual_trans_profile = zeros(N,num_users);
%******THE GOAL OF THIS FUNCTION IS TO FIND AT WHAT TIME ARE THE CHUNKS
%RECEIVED AT THE USERS. NOT ONLY "WHEN" BUT WE ALSO WANT TO FIND OUT "HOW
%MUCH" OF EACH CHUNK IS RECEIVED WHEN DROPPING IS ALLOWED
allchunk_reception_profile = inf*ones(N,num_users); % this is the matrix which gives us the
                                                    % chunk reception times
                                                    % over one sample path
                                                    % for all the users
helper_assign  = zeros(N,num_users); %indicates which helpers have been assigned the current
                                     % request
rrr = cumsum(rr,3);% This 3 dimensional matrix tells us total arrival bits provided to all queues until
                    % every slot in the sample path.

%******THIS FOR LOOP IS TO FIND OUT THE RECEPTION TIME OF THE CHUNKS OVER
%ONE SAMPLE PATH
for i = 1:N-1
   %if(i>50)
    %   zz_bs(4) = 0;
   %end
   
chunk_reception_time = inf*ones(1,num_users);
chunk_list = repmat(rrr(:,:,i),[1,1,N]);%sum of the amount of bits of all chunks that have been 
%requested by a user until time $i$
helper_index = (rr(:,:,i) > 0); %the helper to which chunk requested at time $i$ has been 
%assigned
[row_ind col_ind] = find(helper_index>0);% find the row index of the helper assign incidence
%matrix, i.e., find the helper number to which the chunk requested at time
%$i$ has been assigned
temporary = zeros(1,num_users);
temporary((zz_bs.*(sum(rr(:,:,i),1)>0))==1) = row_ind;%the index inside temporary singles out
%the users (among the active users) who have requested a non  zero amount
%of bits for chunk requested at slot $i$.
helper_assign(i,:) = temporary;%the helper to which chunk requested by an active user at time 
%slot $i$ has been assigned for delivery
chunk_list = chunk_list.*repmat(helper_index,[1,1,N]);% first single out the helpers to which
%chunk at time slot $i$ has been assigned. now, we have the matrix
%chunk_list which keeps track of the sum of the bits that have been
%requested until time $i$ at every queue. multiplying by the "repeated over
%time" helper_index matrix gives us the sum of all bits that have been
%requested until time $i$ at all the helpers that have been assigned chunks
%at time slot $i$. More precisely, every slot $i$, there is a set of
%helpers (call them active queues) which get assigned current requests. 
%Find out the total amount of
%bits that have been assigned till now to only these 'active queues'
served_bits = cumsum(mumu.*repmat(helper_index,[1,1,N]),3);%the list of total amount of service 
% offered until every slot to the 'active queues' at slot $i$.
served_chunks = (served_bits >= chunk_list);
served_chunks = served_chunks.*repmat(helper_index,[1,1,N]);

                %(served_bits < chunk_list)
service_time_index = cumsum(served_chunks,3);



%service_time_index(service_time_index > 1) = 0;
[row,col] = find(service_time_index==1);
quotient = floor(col./num_users);
slot_index = quotient+1;
remainder = mod(col,num_users);
user_index = remainder;
rem = (remainder == 0);
slot_index(rem) = slot_index(rem)-1;
user_index(rem) = num_users;
chunk_reception_time(user_index) = slot_index;
allchunk_reception_profile(i,:) = chunk_reception_time; 
%actual_trans_profile(i,:) = sum(actual_transbits.*helper_index,1); 


i
end
