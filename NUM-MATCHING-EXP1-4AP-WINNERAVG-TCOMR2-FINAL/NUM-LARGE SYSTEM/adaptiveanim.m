%# control animation speed
DELAY = 2/3;
numPoints = 600;
load('qualit4userspsnrcongest.mat','qual_profile');
%# create data
%x = linspace(0,10,numPoints);
%y = log(x);
x = 1:200;
y = qual_profile(:,3);

%# plot graph
figure('DoubleBuffer','on')                  %# no flickering
plot(x,y,'--', 'LineWidth',1), grid on
xlabel('time slot'), ylabel('quality chosen'), title('user 3')

%# create moving point + coords text
hLine = line('XData',x(1), 'YData',y(1), 'Color','r', ...
    'Marker','o', 'MarkerSize',6, 'LineWidth',2);
hTxt = text(x(1), y(1), sprintf('(%.3f,%.3f)',x(1),y(1)), ...
    'Color',[0.2 0.2 0.2], 'FontSize',8, ...
    'HorizontalAlignment','left', 'VerticalAlignment','top');

%# infinite loop
i = 1;                                       %# index
%while (i<=200) 
for i = 1:200
    %# update point & text
    set(hLine, 'XData',x(i), 'YData',y(i))   
    set(hTxt, 'Position',[x(i) y(i)], ...
        'String',sprintf('(%.3f,%.3f)',[x(i) y(i)])) 
    M(i) = getframe(gcf);
 %   drawnow                                  %# force refresh
    pause(DELAY)                           %# slow down animation

    %i = i+1;                %# circular increment
    %if ~ishandle(hLine), break; end          %# in case you close the figure
end
movie(M);