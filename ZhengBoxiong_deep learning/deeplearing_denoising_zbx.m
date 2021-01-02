%运行此代码前需先安装 matconvnet
format compact;
global sigmas; % 输入噪声等级
addpath(fullfile('utilities'));

%定义相关变量
folderModel = 'model';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'temp'};
setTestCur  = imageSets{1};  % 当前测试的数据集
showResult  = 1;
useGPU      = 0; 
pauseTime   = 0;

imageNoiseSigma = 0;  % 设置加噪等级
inputNoiseSigma = 30;  % 设置去噪等级



%载入模型
load(fullfile('model','FDnCNN_gray.mat'));
net = vl_simplenn_tidy(net);


close all;
tic
%输入矩阵，切片处理
for i=1:size(T,3)
    minT = min(T(:));
    maxT = max(T(:));
    label=(T(:,:,i)-minT)/(maxT-minT);
    [w,h,]=size(label);
    
    % 加噪
    randn('seed',0);
    noise = imageNoiseSigma/255.*randn(size(label));
    input = single(label+noise);
    
    
    % 设置噪声水平图
    sigmas = inputNoiseSigma/255; 
    
    % 去噪
    res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet
    %res    = vl_ffdnet_matlab(net, input); % 如果没有安装 matconvnet 请使用这行代码，运行速度很慢 note: you should also comment net = vl_simplenn_tidy(net); and if useGPU net = vl_simplenn_move(net, 'gpu') ; end
    output = res(end).x;
    
    
    figure
    if showResult
        ref = (ref0(:,:,i)-minT)/(maxT-minT);
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(label,output,0,0);%  计算PSNR和SSIM
        disp([num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        imshow(cat(2, ref,label,output, output-label, output-ref),[])
        drawnow;
        pause(pauseTime)
    end
    
end
toc

%SSIM取值范围[0,1]，值越大，表示图像失真越小.
%PSNR的单位是dB，数值越大表示失真越小


