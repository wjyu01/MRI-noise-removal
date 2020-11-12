
%运行此代码前需先安装 matconvnet
format compact;
global sigmas; % 输入噪声等级
addpath(fullfile('utilities'));

%输入矩阵，转换为图片格式
flodertemp = 'denoise temp';
for s=1:size(datass,3)
    a=datass(:,:,s);
    imwrite(im2uint8(a),['C:\deep learning denoise\DnCNN-master\testsets\temp\',num2str(s),'.png']);
    
end

%定义相关变量
folderModel = 'model';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'BSD68','temp'};
setTestCur  = imageSets{2};      % 当前测试的数据集
showResult  = 1;
useGPU      = 0; %不使用GPU
pauseTime   = 0;

imageNoiseSigma = 0;  % 设置加噪等级
inputNoiseSigma = 20;  % 设置去噪等级

%定义去噪结果保存路径
%folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(inputNoiseSigma)]);
folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(imageNoiseSigma),'_',num2str(inputNoiseSigma)]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

%载入模型
load(fullfile('model','FDnCNN_gray.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end


% 图像格式
ext         =  {'*.jpg','*.png','*.bmp'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
end

% PSNR 和 SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)
    
    % 读取图像
    label = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,~]=size(label);
    if size(label,3)==3
        label = rgb2gray(label);
    end
    
    [~,nameCur,extCur] = fileparts(filePaths(i).name);
    label = im2double(label);
    
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
    [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
    if showResult
        %imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));展示加噪后图像、原图、去噪效果图
        imshow(cat(2,im2uint8(label),im2uint8(output)));%仅展示原图和去噪后效果
        title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma,'%02d'),'_' num2str(inputNoiseSigma,'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
        drawnow;
        pause(pauseTime)
    end
    disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
    PSNRs(i) = PSNRCur;
    SSIMs(i) = SSIMCur;
end

disp([mean(PSNRs),mean(SSIMs)]);




