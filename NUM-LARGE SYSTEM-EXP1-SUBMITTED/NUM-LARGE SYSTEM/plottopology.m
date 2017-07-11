load('toplarge.mat','z_h','z_u');
x_h = real(z_h);
y_h = imag(z_h);
x_u = real(z_u);
y_u = imag(z_u);
r = 60;
%****DRAWING CIRCLE USING RECTANGLE() COMMAND******
set(gca,'XTickMode','manual')
set(gca,'YTickMode','manual')

 for i = [1 2]
     xxx = x_h(i) - r;
     yyy = y_h(i) - r;
     rr = 2*r;
     rectangle('Position',[xxx,yyy,rr,rr],'Curvature',[1,1],'Linestyle',':','LineWidth',2);
     hold  all
 end
 i = 13;
 rr = 2*r;
  xxx = x_h(i) - r;
  yyy = y_h(i) - r;
 rectangle('Position',[-280,-280,560,560],'Curvature',[1,1],'Linestyle',':','LineWidth',2);
scatter(x_u(1:4),y_u(1:4),12,'bo');
hold all
scatter(x_h([1 2]),y_h([1 2]),25,'*r');
scatter(x_h(13),y_h(13),100,'r^');
text(x_u(1:4),y_u(1:4),num2cell([1:4]));
set(gca,'XTick',-360:80:360);
set(gca, 'YTick',-360:80:360);
grid(gca)