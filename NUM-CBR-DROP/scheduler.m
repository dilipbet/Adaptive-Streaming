function[mu,mu_bs]=scheduler(z,cap_mat,zz,bs_queue,bs_cap,no_helper,virt_q,virt_bs,zz_bs)
z1 = (z+virt_q).*cap_mat; %the coefficinet in the objective of LP for each helper
z1(zz==0) = -inf; % set the inactive link queue lengths to -inf so that the helpr never 
                  % chooses that user to schedule. Thus, each helper
                  % chooses among only those users which have zz=1 a that
                  % location, ie.e., the connectiviity incidence matrix of
                  % the bipartite graph has a 1 in that location.
temp = max(z1,[],2); % find the user with the best queue*capacity product. 
x = size(z);
best_user_id = (z1 == repmat(temp,1,x(2)));% find the id of the best user
%best_user_id = best_user_id;
tempor = cumsum(best_user_id,2);
best_user_id(tempor>1)=0;% in case of tie, choose the user which comes first on the row list
mu = cap_mat.*best_user_id;% find the rate allocations for the best user
z11 = (bs_queue+virt_bs).*bs_cap;
z11(zz_bs==0) = -inf;
best_bs_id = (z11 == repmat(max(z11,[],2),1,length(bs_queue)));
tempora = cumsum(best_bs_id,2);
best_bs_id(tempora>1) = 0;
mu_bs = bs_cap.*best_bs_id;% find the rate allocation for the best user
if(no_helper)
    mu = zeros(x(1),x(2));% if there are no helpers, then don't assign any rates to the users
                          % from helpers.
end
%num_best_users = sum(best_user_id,2);
%best_user_id.*repmat(num_best_users.*(num_best_users > 1),1,x(2));
 



