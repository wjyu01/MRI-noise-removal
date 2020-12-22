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
useGPU      = 0; %不使用GPU
pauseTime   = 0;

imageNoiseSigma = 0;  % 设置加噪等级
inputNoiseSigma = 40;  % 设置去噪等级

%定义去噪结果保存路径
%folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(inputNoiseSigma)]);
folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(imageNoiseSigma),'_',num2str(inputNoiseSigma)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

%载入模型
load(fullfile('model','FDnCNN_gray.mat'));
net = vl_simplenn_tidy(net);

% 图像格式
 %ext         =  {'*.jpg','*.png','*.bmp'};
 %filePaths   =  [];
 


% PSNR 和 SSIM
 %PSNRs = zeros(1,length(filePaths));
 %SSIMs = zeros(1,length(filePaths));

% for i = 1:length(filePaths)
%[~,nameCur,extCur] = fileparts(filePaths(i).name);
close all;
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
    
    % toc;
    % 计算 PSNR, SSIM，并保存结果
     %[PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    figure
    if showResult
        ref = (T(:,:,i)-minT)/(maxT-minT);
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(label,output,0,0);
        disp([num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        imshow(cat(2, ref,label,output, output-label, output-ref),[])
        %imwrite(output, fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma,'%02d'),'_' num2str(inputNoiseSigma,'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ))
        drawnow;
        pause(pauseTime)
    end
    
    %disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
    %PSNRs(i) = PSNRCur;
    %SSIMs(i) = SSIMCur;
end

%disp([mean(PSNRs),mean(SSIMs)]);
%SSIM取值范围[0,1]，值越大，表示图像失真越小.
%PSNR的单位是dB，数值越大表示失真越小


