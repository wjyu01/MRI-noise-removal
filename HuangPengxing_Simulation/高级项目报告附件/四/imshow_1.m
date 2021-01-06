
  K = reshape(VImg.Sx + i * VImg.Sy, 300, 300, 2);
  result = abs(fftshift(ifft(ifft2(K),[],3)));
% result = abs(fftshift(ifft2(K)));
for n = 1:2
    K_n = result(:,:,n);
    figure;
    imshow(K_n,[])
end

for n = 1:2
    K_n = K(:,:,n);
    figure;
    imshow(K_n,[])
end





