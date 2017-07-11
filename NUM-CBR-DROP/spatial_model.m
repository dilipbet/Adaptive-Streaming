function[z_h,z_u] = spatial_model(macro_side,cell_side,helpers_per_cell,users_per_cell)
%global macro_side;
%global cell_side;
% We assume a 400m X 400m grid with 25 cells. Each cell is an 80m X 80m
% square. The BS is at the origin
%macro_side = 400; % side length of macro square grid
%cell_side = 100; % side length of cell square
num_cells = (macro_side/cell_side)^2; %total number of cells
%helpers_per_cell = 3;
%users_per_cell = 2;
x_min = -floor(macro_side/(2*cell_side));% minimum poosible x-coordinate of center of a cell
y_min = x_min; %min possible y-coordinate of center of a cell
x_max = -x_min - 1; %max possible x-coordinate of centre of cell
y_max = x_max - 1; % max possible y-coordinate of center of cell

%****x-coordinates of helpers********
cell_xcoord = ones(num_cells*helpers_per_cell,1);
cell_xcoord(1,1) = 0;
cell_xcoord = cumsum(cell_xcoord);
cell_xcoord = floor(cell_xcoord/helpers_per_cell);
cell_xcoord = mod(cell_xcoord,sqrt(num_cells));
%x_h = unifrnd(-cell_side/2,cell_side/2,num_cells*helpers_per_cell,1);%for random placement of
x_h = 0; %for helpers placed at the center of each cell                                                                     % helpers
x_h = x_h + cell_side*(cell_xcoord + x_min);
%x_h = x_h + 20
%*****y-coordinates of helpers*******
cell_ycoord = ones(num_cells*helpers_per_cell,1);
cell_ycoord(1,1) = 0;
cell_ycoord = cumsum(cell_ycoord);
cell_ycoord = floor(cell_ycoord/(sqrt(num_cells)*helpers_per_cell));
%y_h = unifrnd(-cell_side/2,cell_side/2,num_cells*helpers_per_cell,1);
y_h = 0;
y_h = y_h + cell_side*(cell_ycoord + y_min);
%y_h = y_h + 20
z_h = complex(x_h,y_h);

%*****x-coordinates of users*****
cell_xcoord_user = ones(1,num_cells*users_per_cell);
cell_xcoord_user(1,1) = 0;
cell_xcoord_user = cumsum(cell_xcoord_user,2);
cell_xcoord_user = floor(cell_xcoord_user/users_per_cell);
cell_xcoord_user = mod(cell_xcoord_user,sqrt(num_cells));
x_u = unifrnd(-cell_side/2,cell_side/2,1,num_cells*users_per_cell);
x_u = x_u + cell_side*(cell_xcoord_user + x_min);
%x_u = x_u + 20;
%*****y-coordinates of users*****
cell_ycoord_user = ones(1,num_cells*users_per_cell);
cell_ycoord_user(1,1) = 0;
cell_ycoord_user = cumsum(cell_ycoord_user);
cell_ycoord_user = floor(cell_ycoord_user/(sqrt(num_cells)*users_per_cell));
y_u = unifrnd(-cell_side/2,cell_side/2,1,num_cells*users_per_cell);
y_u = y_u + cell_side*(cell_ycoord_user + y_min);
%y_u = y_u + 20;
z_u = complex(x_u,y_u);
num_helpers = num_cells*helpers_per_cell;
r = 60;
%*******DRAWING CIRCLE USING PLOT******
a = linspace(0,2*pi,10);
xx = repmat(x_h,1,length(a)) + repmat(r*cos(a),num_helpers,1);
yy = repmat(y_h,1,length(a)) + repmat(r*sin(a),num_helpers,1);
size(xx)
size(yy)

%****DRAWING CIRCLE USING RECTANGLE() COMMAND******
set(gca,'XTickMode','manual')
set(gca,'YTickMode','manual')

 for i = 1:num_helpers
     xxx = x_h(i) - r;
     yyy = y_h(i) - r;
     rr = 2*r;
     rectangle('Position',[xxx,yyy,rr,rr],'Curvature',[1,1],'Linestyle',':','LineWidth',1);
     hold  all
 end
 
scatter(x_u,y_u,'b*');
hold all
scatter(x_h,y_h,'*r');
set(gca,'XTick',-360:80:360);
set(gca, 'YTick',-360:80:360);
grid(gca)