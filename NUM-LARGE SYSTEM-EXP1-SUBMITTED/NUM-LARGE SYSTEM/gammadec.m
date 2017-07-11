function[gamma] = gammadec(theta,zz_bs,V)
gamma = V./theta;
gamma(gamma>1) = 1;
gamma(gamma<0.4633) = 0.4633;
gamma = gamma.*zz_bs;