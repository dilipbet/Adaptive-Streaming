DELAY = 2/3;
bitperchunk = xx(1:1600,3:4);
i = 0:199;
figure('DoubleBuffer','on')  
%subplot(2,1,1);
 for j = 1:8
layer = bitperchunk(8*i+j,2);
grid on
plot(layer*1.5/1000)
hold all
end
hold all
x = 1:200;
yy1 = 1.5*received_chunkbits(:,1)/1000;
plot(x,yy1,'--black', 'LineWidth',2);
title ('adaptive streaming for user 1')
xlabel('slot number (1 slot = 1 GOP duration)')
ylabel('bitrate (in kbps) for 8 layers of encoded video')
thruput_bs = permute(offered(26,1,:),[1 3 2]);
thruput_helper = permute(offered(1:2,1,t2+1:200),[1 3 2]);
thruput_helper = max([thruput_helper;thruput_bs(t2+1:200)]);
%reception_rate(1:35) = cumsum(thruput_bs)./cumsum(1:35);%
%reception_rate(1:30) = prev_bscap/4;
%reception_rate(31:35) = cap_bs(1)/4;
%reception_rate(36:200) = cumsum(thruput_helper)./(cumsum(1:165));
%reception_rate(36:100) = max([cap_mat(1,1)/2 cap_mat(2,1)/3]);
%reception_rate(101:200) = max([cap_mat(1,1)/2 cap_mat(2,1)/2]);
reception_rate(1:t1) = mean(thruput_bs(1:t1));
reception_rate(t1+1:t2) = mean(thruput_bs(t1+1:t2));
reception_rate(t2+1:t3) = mean(thruput_helper(1:t3-t2));
reception_rate(t3+1:200) = mean(thruput_helper(t3-t2+1:200-t3+t2));
yy2 = reception_rate*1.5/1000;
%subplot(2,1,2);

% plot(x,yy2);
% title('average rate offered by the system to user 1 under different conditions')
% xlabel('slot number')
% ylabel('average rate offered by the system (in kbps)')
% grid on


%# create 2 moving points + coords text
hLine1 = line('XData',x(1), 'YData',yy1(1), 'Color','r', ...
    'Marker','o', 'MarkerSize',6, 'LineWidth',2);
hTxt1 = text(x(1), yy1(1), sprintf('(%.3f,%.3f)',x(1),yy1(1)), ...
    'Color',[0.2 0.2 0.2], 'FontSize',8, ...
    'HorizontalAlignment','left', 'VerticalAlignment','top');


% hLine2 = line('XData',x(1), 'YData',yy2(1), 'Color','g', ...
%     'Marker','o', 'MarkerSize',6, 'LineWidth',2);
% hTxt2 = text(x(1), yy2(1), sprintf('(%.3f,%.3f)',x(1),yy2(1)), ...
%     'Color',[0.2 0.2 0.2], 'FontSize',8, ...
%     'HorizontalAlignment','left', 'VerticalAlignment','top');

%# infinite loop
i = 1;                                       %# index
%while (i<=200) 
for i = 1:200
    %# update point & text
    set(hLine1, 'XData',x(i), 'YData',yy1(i))   
    set(hTxt1, 'Position',[x(i) yy1(i)], ...
       'String',sprintf('(%.3f,%.3f)',[x(i) yy1(i)])) 
%     set(hLine2, 'XData',x(i), 'YData',yy2(i))   
%     set(hTxt2, 'Position',[x(i) yy2(i)], ...
%         'String',sprintf('(%.3f,%.3f)',[x(i) yy2(i)])) 
    M1(i) = getframe(gcf);
 %   drawnow                                  %# force refresh
    %pause(DELAY)                           %# slow down animation

    %i = i+1;                %# circular increment
    %if ~ishandle(hLine), break; end          %# in case you close the figure
end
%movie(M);
