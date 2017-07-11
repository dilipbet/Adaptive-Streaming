function[gamma] = gammadec(theta,zz_bs,V,active_users)
gamma = V./theta;
gamma(gamma>1) = 1;
gamma(gamma<0.4633) = 0.4633;
gamma = gamma.*active_users;