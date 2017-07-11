bitperchunk = xx(:,3:4);
i = 0:799;
%subplot(2,1,1);
for j = 1:8
layer = bitperchunk(8*i+j,2);
poof = circshift(layer, [0 -16]);
grid on
plot(poof/1000)
hold all
end

