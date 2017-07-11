bitperchunk = xx(1:1600,3:4);
i = 0:199;
subplot(2,1,1);
for j = 1:8
layer = bitperchunk(8*i+j,2);
grid on
plot(layer*1.5/1000)
hold all
end
hold all
plot(1.5*received_chunkbits(:,1)/1000);
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

subplot(2,1,2);

plot(reception_rate*1.5/1000)
title('average rate offered by the system to user 1 under different conditions')
xlabel('slot number')
ylabel('average rate offered by the system (in kbps)')
grid on
