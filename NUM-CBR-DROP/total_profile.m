i = 1;
chunk_delay = allchunk_reception_profile(1:N-3,i) - cumsum(ones(N-3,1),1);
vec1 = received_chunkbits(1:N-3,i);
vec2 = layer_evolve(1:N-3,i);
vec3 =  dropped_chunkbits(1:N-3,i);
[vec2 vec1 vec3 chunk_delay helper_assign(1:N-3,i)]


subplot(3,1,1);
%helper_assign(helper_assign==26) = 3;
plot(helper_assign(1:N-3,1))
xlabel('chunk number')
ylabel('index of helper')
subplot(3,1,2);
plot(chunk_delay)
 xlabel('chunk number')
 ylabel('delay in removal')
 subplot(3,1,3);
 plot(vec1>0, 'o')
 xlabel('chunk number')
 ylabel('1 = delivered 0 = not yet delivered')
 alluser_chunk_delay = allchunk_reception_profile(:,1:4) - repmat(cumsum(ones(N,1),1),1,4);