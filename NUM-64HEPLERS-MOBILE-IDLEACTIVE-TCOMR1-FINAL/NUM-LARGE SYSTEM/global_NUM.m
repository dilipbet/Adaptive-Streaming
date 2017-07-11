clear all
users_per_cluster=2;
helpers_per_cluster = 2; 
deg_user = 2;
deg_helper = 2;
num_clusters = 2;
% thus, we have 15 queues per cluster. The rates achievable in this cluster
% depends on the interference seen from surrounding clusters. Distortion
% corresponds to each user. Thus, there will be (5*number of clusters)
% users in the whole area. We can assume that the 5 users are uniformly
% distributed in the area of that cluster or almost equidistant from all
% the helpers in the cluster. each helper is allocated different frequency
% and it time shares among the 5 users. thus, there are 3 frequencies.
% further, each receiver sees interference on all the 3 frequencies from 2
% or 3 neighboring base stations.
delta = 40;
alpha = 3.5;
user_helper_dist = 40;
num_interferers = 1;
user_interferer_dist = 80;
femto_power = 10^(2.3);
BS_power = 10^(4.3);
pl_helper = 1./(1+(user_helper_dist/delta).^alpha);
pl_interf = 1./(1+(user_interferer_dist/delta).^alpha);
SINR = (femto_power*pl_helper)/...
    (num_interferers*femto_power*pl_interf);
p2p_capacity = log2(1+SINR);
z_cluster = ones(helpers_per_cluster,users_per_cluster); % queues at helpers in a cluster
                               %heleprs in a cluster operate on different
                               %frequencies
zz = z_cluster;
for i = 1:num_clusters-1
zz = blkdiag(zz,z_cluster);
end
z =zz;
size_vec = size(z);
z(z>0) = 0;
V = 100;
cap_mat = repmat(p2p_capacity,size(z));
%cap_mat = repmat(p2p_capacity,size(z));
N = 100;
util = zeros(1,N);
util2 = zeros(1,N);
q_evolve = zeros(size_vec(1),size_vec(2),N);
d_evolve = zeros(1,size_vec(2),N);
    %****** QUEUE UPDATE****
    %[r,d_choice] = congestion_control(z,V,zz);
    %z = z + r;
    
for i = 1:N
    [r,d_choice] = congestion_control(z,V,zz);
    mu = scheduler(z,cap_mat,zz);
    z = z + r - mu;
    r
    d_choice
    mu
    z(z<0) = 0;
    z
    %qq = sum(sum(z))
    q_evolve(:,:,i) = z;
    d_evolve(:,:,i) = d_choice;
    util (i) = sum(d_choice,2);
    util2(i) = sum(sum(z));
end
%plot(cumsum(util)./cumsum(ones(1,N)));


