function[mu,mu_bs,exitflag]=scheduler(z,cap_mat,zz,bs_queue,bs_cap,no_helper,zz_bs,...
    num_helpers,num_users,A,b,edge_indices)
z1 = ((z).*cap_mat).*zz; %the coefficinet in the objective of LP for each helper
incidence_dim = size(A);
weights = zeros(incidence_dim(2),1);
for indind = 1:incidence_dim(2)
weights(indind) = z1(edge_indices(indind,1),edge_indices(indind,2));
end
[alpha,primalval,exitflag] = linprog(-weights/(10^13),A,b,[],[],zeros(incidence_dim(2),1));

mu = zeros(num_helpers,num_users);
for indind = 1:incidence_dim(2)
    mu(edge_indices(indind,1),edge_indices(indind,2)) = alpha(indind);
end
 mu = mu.*cap_mat;
 mu_bs = zeros(1,num_users);



