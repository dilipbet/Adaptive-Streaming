function[gamma] = gammadec(theta,zz_bs,V)
gamma = V./theta - 1;
gamma(gamma>1) = 1;
gamma(gamma<0.8595) = 0.8595;
gamma = gamma.*zz_bs;