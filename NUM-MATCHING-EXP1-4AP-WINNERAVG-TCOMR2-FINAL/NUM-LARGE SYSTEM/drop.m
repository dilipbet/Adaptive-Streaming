function[dropped,dropped_bs]=drop(z,virt_q,V,beta,bs_queue,virt_bs,R_max,num_helpers,num_users,zz,zz_bs)
dropped = zeros(num_helpers,num_users);
greater = (z+virt_q)>V*beta;
dropped(greater) = R_max;

dropped(~greater) = 0;

greater_bs = (bs_queue+virt_bs)>V*beta;
dropped_bs(greater_bs) = R_max;
dropped_bs(~greater_bs) = 0;
dropped = dropped.*zz;
dropped_bs = dropped_bs.*zz_bs;
