bitperchunk = xx(:,3:4);
i = 0:799;
%subplot(2,1,1);
for j = 1:8
layer = bitperchunk(8*i+j,2);
grid on
plot(layer)
hold all
end

