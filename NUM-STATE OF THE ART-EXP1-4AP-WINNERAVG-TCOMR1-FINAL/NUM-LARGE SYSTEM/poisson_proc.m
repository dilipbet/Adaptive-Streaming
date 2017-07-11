function pts=poisson_proc(X,Y,lamda)
if Y==0
    %%% generate 1D poisson process
    n=random('poisson',X*lamda);  %The number of points to generate
    pts=rand(n,1);
    pts(:,1)=pts(:,1)*X;
    pts=sort(pts);    
else
    %%% generate 2D poisson process
    n=random('poisson',X*Y*lamda);  %The number of points to generate
    pts=rand(n,2);
    pts(:,1)=pts(:,1)*X;
    pts(:,2)=pts(:,2)*Y;
end
end