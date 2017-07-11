
load('topologytheatre5.mat','z_h','z_u');
x_h = real(z_h);
y_h = imag(z_h);
x_u = real(z_u);
y_u = imag(z_u);
figure(10) 
scatter(x_u,y_u,'b*');
hold all
scatter(x_h,y_h,'*r');
scatter(0,0,'^g');
%set(gca,'XTick',-360:80:360);
%set(gca, 'YTick',-360:80:360);
grid(gca)