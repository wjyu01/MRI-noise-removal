x=result; %result 更改为变量数据名称
R=real(x);
[M,N,Y]=size(R);
b=max(R(:));
c=min(R(:));
R=(R-c)/(b-c);
g=imnoise(R,'salt & pepper',0.05);

X=imag(x);
[M,N,U]=size(X);
d=max(X(:));
e=min(X(:));
X=(X-e)/(d-e);
f=imnoise(X,'salt & pepper',0.05);
T=(g*(b-c)+c)+((f*(d-e)+e)*1i);

subplot(121),montage(reshape(abs(x),M,N,1,[]),'displayrange',[]),title('原图像');

subplot(122),montage(reshape(abs(T),M,N,1,[]),'displayrange',[]),title('添加椒盐噪声图像');