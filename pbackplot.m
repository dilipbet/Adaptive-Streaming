k = [1 2 3 4 5 6 7 8 9 10 11 12 13];
a = [3 4 6 9 10 11 12 13 16 17 19 20 21];
b = zeros(1,length(k));
for i = 1 : length(a)
b(i) = max(a(1:i));
end
pb(1) = 0;
consump = zeros(1,length(k));
prebuff = 9;
pb = zeros(1, length(k));
for j = 2: length(a)
    if(j>prebuff)
   consump(j) = consump(j-1)+1;
    else
        consump(j) = consump(j-1);
    end
pb(j) = pb(j-1)+sum(b==j);
end
stairs(pb)
hold all
stairs(consump)
grid on
set(gca, 'xtick',[1:1:length(a)]);
set(gca, 'ytick', [1:1:length(a)]);
text(4.5, 0.75, 'd', 'Color','r','FontSize',14)
xlabel('video chunk slot (i)','FontSize',14)
ylabel('number of chunks','FontSize',14)
ax = legend('no. of chunks in playback buffer','no.of chunks consumed by playback')
LEG = findobj(ax,'type','text');
set(LEG,'FontSize',14)