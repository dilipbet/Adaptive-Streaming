function[BS_locations, user_locations,num_helpers,num_users,r] = BS_user_placement
%%********This function places base stations in whichever way you want*****
%num_cells = macro_side/cell_side; % for simple 1-D model
%num_cells = Nx*Ny;
%[Lxy_t,Lxy,X,Y] = arch_layout(M,Nx,Ny);
% small cell base stations are located at the centre of every voronoi cell
%****positions of femto BS's *****
%[Dab] = modulo_distance(Lxy_t,Lxy,X,Y);
%BS_locations = reshape(Lxy_t.',1,Nx*Ny);      

 BS_locations = [40;120] + sqrt(-1)*[40;40]; %for simple 1-D example
%****** USER POSITIONS*************
%user_locations = Lxy.';
%***** Users are generated from either a uniform distribution or Gaussian
%distribution*****
user_locations(1:50) = unifrnd(20,80,1,50) + sqrt(-1)*unifrnd(30,60,1,50);
%user_locations(91:100) = unifrnd(0,160,1,10)+ sqrt(-1)*unifrnd(0,80,1,10);
%user_locations(1:150) = linspace(1,10,150);
%user_locations(151:200) = linspace(40,79,50);

%user_locations = 20*ones(1,num_users); % simple 1-D model
%user_locations(1:num_users - 10) = 5;
%user_locations(num_users-9:num_users) = 35;
r = 60;
num_users = 50;
num_helpers = 2;
x_h = real(BS_locations);
y_h = imag(BS_locations);
x_u = real(user_locations);
y_u = imag(user_locations);
%set(gca,'XTickMode','manual')
%set(gca,'YTickMode','manual')

 for i = 1:length(BS_locations)
     xxx = x_h(i) - r;
     yyy = y_h(i) - r;
     rr = 2*r;
     rectangle('Position',[xxx,yyy,rr,rr],'Curvature',[1,1],'Linestyle',':','LineWidth',1);
     hold  all
 end
 
scatter(real(user_locations),imag(user_locations),'b*');
hold all
scatter(real(BS_locations),imag(BS_locations),'*r');
%set(gca,'XTick',-360:80:360);
%set(gca, 'YTick',-360:80:360);
%grid(gca)




