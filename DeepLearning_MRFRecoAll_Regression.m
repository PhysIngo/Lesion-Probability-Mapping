%% Prepare all the parameters and datastore for training
% Folder = 'F:\users\ih11\DeepLearning MRFMS\MRFReco_Full\';
Folder = 'F:\users\ih11\DeepLearning MRFMS\MRFReco_Full\';

inputFolder = 'RotMRF';
% inputFolder = 'RotT1T2star';

% outputFolder = 'RotLesions';
outputFolder = 'RotT1T2starMaskLesions';


maxEpochs = 100;
isPatches = true;

losses = 'mse';

% this is the training dataset
inputData = imageDatastore(fullfile(Folder,inputFolder),'FileExtensions',...
    '.mat','ReadFcn',@matRead);
outputData = imageDatastore(fullfile(Folder,outputFolder),'FileExtensions',...
    '.mat','ReadFcn',@matRead);


k = size(inputData.Files,1); % number of folds
% k = 5;
% partStores{k} = [];
for i = 1:k
   temp = partition(inputData, k, i);
   inputPartStores{i} = temp.Files;
%    inputPartStores{i} = shuffle(inputPartStores{i});
   
   temp = partition(outputData, k, i);
   outputPartStores{i} = temp.Files;
%    ouputPartStores{i} = shuffle(ouputPartStores{i});
end

idx = crossvalind('Kfold', k, k);
% idx = 1:k;

% generate the UNet with inputDim and outputDim;

if isPatches
    PU = 'gpu';
    initLearningRate = 10^-4;%10^-4
    dim = 64;
    patchis = 24;
    miniBatchSize = 64;
    pF = 1;
else
    PU = 'gpu';
    initLearningRate = 0.001;
    dim = 240;
    patchis = 1;
    miniBatchSize = 32;
    pF = 14;
end

saveFold = ['CheckFolder_',PU,'_',losses,'_patches_',outputFolder,'100'];

mkdir([Folder,saveFold])

tmp = readimage(inputData,1);
inputDim = size(tmp,3);
tmp = readimage(outputData,1);
outputDim = size(tmp,3);
generateLayers = true;
if generateLayers %
    lgraph = unetLayers([dim dim inputDim] , 5,'encoderDepth',3);
    lgraph = lgraph.removeLayers('Softmax-Layer');
    lgraph = lgraph.removeLayers('Segmentation-Layer');
    lgraph = lgraph.removeLayers('Final-ConvolutionLayer');
    lgraph = lgraph.addLayers(convolution2dLayer(1,outputDim,'name','Final-ConvolutionLayer'));
    if isempty(losses)
        lgraph = lgraph.addLayers(regressionLayer('name','myRegressionLayer'));
    else
        lgraph = lgraph.addLayers(RegressionLayer_Function(losses));
    end
    %lgraph = lgraph.connectLayers('Decoder-Stage-3-ReLU-2','Final-ConvolutionLayer');
    lgraph = lgraph.connectLayers('Decoder-Stage-4-ReLU-2','Final-ConvolutionLayer');
    lgraph = lgraph.connectLayers('Final-ConvolutionLayer','myRegressionLayer');
%     figure;plot(lgraph);
    

%     analyzeNetwork(lgraph)
end


    
% reset(gpuDevice)
% g=gpuDevice([]);

modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');

all_names = inputData.Files;
for ti = 1:1:size(inputData.Files,1)
test_names = {'a11','a20','a29','b10','b17'};
valid_names = {'a06','a13','a23','b11','b08'};
    theName = all_names{ti};
    for tc = 1:1:size(test_names,2)
        if contains(theName,test_names{tc})
            test_idx(ti) = logical(1);
            break;
        else
            test_idx(ti) = logical(0);
        end
    end
    
    for tc = 1:1:size(valid_names,2)
        if contains(theName,valid_names{tc})
            valid_idx(ti) = logical(1);
            break;
        else
            valid_idx(ti) = logical(0);
        end
    end
end
train_idx = ~test_idx & ~valid_idx;


input_test_Store = imageDatastore(cat(1, inputPartStores{test_idx}), 'FileExtensions',...
'.mat','ReadFcn',@matRead);
input_valid_Store = imageDatastore(cat(1, inputPartStores{valid_idx}), 'FileExtensions',...
'.mat','ReadFcn',@matRead);
input_train_Store = imageDatastore(cat(1, inputPartStores{train_idx}), 'FileExtensions',...
'.mat','ReadFcn',@matRead);

output_test_Store = imageDatastore(cat(1, outputPartStores{test_idx}), 'FileExtensions',...
'.mat','ReadFcn',@matRead);
output_valid_Store = imageDatastore(cat(1, outputPartStores{valid_idx}), 'FileExtensions',...
'.mat','ReadFcn',@matRead);
output_train_Store = imageDatastore(cat(1, outputPartStores{train_idx}), 'FileExtensions',...
'.mat','ReadFcn',@matRead);

testData=combine(input_test_Store,output_test_Store);
validData=combine(input_test_Store,output_test_Store);
trainData=combine(input_train_Store,output_train_Store);

if isPatches
    patch_train = randomPatchExtractionDatastore(input_train_Store,output_train_Store,dim, ...
         'PatchesPerImage',patchis);
     trainData = patch_train;

    patch_test = randomPatchExtractionDatastore(input_test_Store,output_test_Store,dim, ...
         'PatchesPerImage',patchis);
     testData = patch_test;

    patch_valid = randomPatchExtractionDatastore(input_valid_Store,output_valid_Store,dim, ...
         'PatchesPerImage',patchis);
     validData = patch_valid;
     validData = patch_test;
end

%'DataAugmentation',augmenter
% augmenter = imageDataAugmenter('RandRotation',[-10 10],'RandXReflection',true);
% augTrainData = augmentedImageDatastore(imageSize,input_train_Store,output_train_Store,'DataAugmentation',augmenter);

options = trainingOptions('adam', ...
    'InitialLearnRate',initLearningRate, ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Plots','training-progress', ...
    'Verbose',true,...
    'Shuffle','every-epoch',...
    'ValidationFrequency',25*patchis,...
    'ValidationData',validData,...
    'ExecutionEnvironment',PU,...
    'GradientThreshold',10^10,...
    'CheckpointPath',[Folder,saveFold]);%,... 
%     'OutputFcn',@(info)stopIfAccuracyNotImproving(info,25)); 


%% train the network and calculate the RMSE
%warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
% try
%     nnet.internal.cnngpu.reluForward(1);
% catch ME
% end

sprintf(['Starting with the Training!']);
isPredict = false;

if ~isPatches
    net = trainNetwork(trainData, lgraph, options);
else
    net = trainNetwork(patch_train, lgraph, options);
end

% end
% end

% save(['CNN-',saveFold,'-maxEpochs-',num2str(maxEpochs),'-',modelDateTime,'.mat'],'net');
%     net{i} = trainNetwork(patchds, lgraph, options);

% calculate the predictions and the RMSE

% save(['FiveOutputMask-maxEpochs-',num2str(maxEpochs),'-',modelDateTime,'.mat'],'net');

set(findall(findall(0,'type','figure'),'type','Axes',...
'Tag','NNET_CNN_TRAININGPLOT_AXESVIEW_AXES_REGRESSION_RMSE'),'YScale','lin','YLim',pF.*[0 3*10^4]);

set(findall(findall(0,'type','figure'),'type','Axes',...
'Tag','NNET_CNN_TRAININGPLOT_AXESVIEW_AXES_REGRESSION_LOSS'),'YScale','lin','YLim',pF.*[0 3*10^8]);
h=findall(groot,'Type','Figure');
h(1,1).MenuBar = 'figure';

allFigures = findall(0,'type','figure');
lossAxes = findall(allFigures,'type','Axes',...
'Tag','NNET_CNN_TRAININGPLOT_AXESVIEW_AXES_REGRESSION_LOSS');
DataArray.Loss.x = lossAxes(1,1).Children(5,1).XData;
DataArray.Loss.y = lossAxes(1,1).Children(5,1).YData;
DataArray.LossSmoothed.x = lossAxes(1,1).Children(4,1).XData; 
DataArray.LossSmoothed.y = lossAxes(1,1).Children(4,1).YData;
DataArray.LossValid.x = lossAxes(1,1).Children(3,1).XData; 
DataArray.LossValid.y = lossAxes(1,1).Children(3,1).YData;
rmseAxes = findall(allFigures,'type','Axes',...
'Tag','NNET_CNN_TRAININGPLOT_AXESVIEW_AXES_REGRESSION_RMSE');
DataArray.RMSE.x = rmseAxes(1,1).Children(5,1).XData;
DataArray.RMSE.y = rmseAxes(1,1).Children(5,1).YData;
DataArray.RMSESmoothed.x = rmseAxes(1,1).Children(4,1).XData; 
DataArray.RMSESmoothed.y = rmseAxes(1,1).Children(4,1).YData;
DataArray.RMSEValid.x = rmseAxes(1,1).Children(3,1).XData; 
DataArray.RMSEValid.y = rmseAxes(1,1).Children(3,1).YData;
DataArray.Name = allFigures(1,1).Name;
save(['DataArray_',DataArray.Name(20:end-10),'_',strrep(DataArray.Name(end-8:end-1),':','-')],'DataArray')

figure;hold on;
plot(DataArray.Loss.x,DataArray.Loss.y);
plot(DataArray.LossValid.x,DataArray.LossValid.y);

%% deep learning maps over different epochs
mP = 25;
S = dir([Folder,saveFold]);
S(1:2,:) = [];
S = S(~[S.isdir]);
[~,idx] = sort(([S.datenum]));
S = S(idx);
isLesions = false;
isDiff = true;

num = 454;
vec = [1,5,40,size(S,1)];
amount = size(vec,2);
im_true = readimage(outputData,num);

if isLesions
    imLes = im_true(:,:,5);
end
dims = size(im_true,3);
clear showImg;
for j=1:1:dims
    showImg{j} = squeeze(im_true(:,:,j));
end
for i=1:1:amount
    newStr = strtrim(S(vec(i),:).name);
    tmpStr = strsplit(newStr,'__');
    load([Folder,saveFold,'\',newStr]);
    iter(i) = vec(i);
    if isPatches
        im_pred = predictPatches(net,readimage(inputData,num),'patchSize',dim);
    else
        im_pred = predict(net, readimage(inputData,num));
    end
    im_pred(im_pred<0) = -im_pred(im_pred<0);

    
    for j=1:1:dims
        showImg{j} = [showImg{j} im_pred(:,:,j)];
    end
end
    
names = {'T_1 [ms]','T_2* [ms]','WM','GM','Lesions'};
t1 = 30;t2 = 30;
figure;
for j=1:1:dims
    subplot(dims,1,j);
    if j==1
        imagesc(showImg{j}.*10,[0 2500]);c=colorbar;
    else
        imagesc(showImg{j},[0 250]);c=colorbar;
    end
    if isLesions && j==5
        hold on;
        h=findobj(gcf,'type','axes');
        h(j,1).YDir = 'reverse';
        LesionKE = SeparateLesions(imLes);
        for k=1:1:max(unique(LesionKE(:)))
            for cc=1:1:amount+1
                polyLes = mask2poly(LesionKE==k,'Exact');
                plot(polyLes(2:end,1)+size(im_true,2)*(cc-1),polyLes(2:end,2),...
                    'linewidth',0.5,'Color',[0.8 0.2 0.1]);
            end
        end
    end
    ylabel(c,names{j});
    text(t1,t2,'true','Color','White');
    for i=1:1:amount
        text(i*240+t1,t2,num2str(iter(i)),'Color','White');
    end
    axis off; axis image;
end


if isDiff
    
for j=1:1:dims
    trueImg{j} = repmat(squeeze(im_true(:,:,j)),1,amount+1);
    binImg = trueImg{j};
    binImg(binImg>0) = 1;
	diffImg{j} = binImg.*(100.*abs(showImg{j}-trueImg{j})./(trueImg{j}));
end
figure;
for j=1:1:dims
    subplot(dims,1,j);
    imagesc(diffImg{j},[0 mP]);c=colorbar;colormap(hot);
    
    
    if isLesions
        hold on;
        h=findobj(gcf,'type','axes');
        h(j,1).YDir = 'reverse';
        LesionKE = SeparateLesions(imLes);
        for k=1:1:max(unique(LesionKE(:)))
            for cc=1:1:amount+1
                polyLes = mask2poly(LesionKE==k,'Exact');
                plot(polyLes(2:end,1)+size(im_true,2)*(cc-1),polyLes(2:end,2),...
                    'linewidth',0.5,'Color',[0.8 0.1 0.2]);
            end
        end
    end
    ylabel(c,names{j});
    text(t1,t2,'true','Color','White');
    for i=1:1:amount
        text(i*dim+t1,t2,num2str(iter(i)),'Color','White');
    end
    axis off; axis image;
end

end


%% generate the hidden layers output
%look at layers
% net.Layers
isSave = false;

clear features_entire newfeatures newfeatures5 allFeatureData

for LayerNum=55   
cc = 1;
clear features_entire
for slcs=597:745
    if (contains(inputData.Files(slcs),'a13') && contains(inputData.Files(slcs),'_00.mat'))

        try
            newfeatures = predictHiddenPatches(net,readimage(inputData,slcs),LayerNum,'patchSize',dim);
        catch
            continue;
        end
        for i=1:1:size(newfeatures,3)
            m1 = meanzeros(meanzeros(newfeatures(:,:,i),1),2);
            mm = m1;
        end

        if isSave
            fig = figure;set(fig, 'Color', [1 1 1], 'Units', 'normalized', 'Position', [0 0 1 1])
            subplot(1,2,1);imagesc(MosaicOnOff(newfeatures5),[0 3]);title(LayerNum);axis off;axis image;
            subplot(1,2,2);imagesc(MosaicOnOff(newfeatures),[0 3]);title(LayerNum);axis off;axis image;
            saveas(gcf,['NetCompare_',num2str(LayerNum),'.png'])
            close(fig);
        end
        features_entire(:,:,:,cc) = newfeatures;
        cc = cc+1;
    end     % end of contains

end     %Layer or slcs loop

end

%% plot all slices of the feature number 15
theNum = [15];
pos = [140 140 32];
tmp = features_entire;

figure;
subplot(2,2,1);imagesc(squeeze(meanzeros(tmp(pos(1),:,theNum,:),3)));axis off;axis image;
subplot(2,2,2);imagesc(squeeze(meanzeros(tmp(:,pos(2),theNum,:),3)));axis off;axis image;
subplot(2,2,3);imagesc(squeeze(meanzeros(tmp(:,:,theNum,pos(3)),3)));axis off;axis image;
subplot(2,2,4);s1=slice(squeeze(meanzeros(tmp(:,:,theNum,:),3)),[pos(1)],[pos(2)],[pos(3)]);
set(s1,'edgecolor','none')

%% Calculate the Predictions for all different Epochs and only for testing or both

losses = 'mse';
add = '';
add2 = '';
isPatches = true;

minNum = 1;maxNum=100;
isSave = true;
isOnlyTrain = true;

allFiles = dir([Folder,inputFolder]);
allFiles = allFiles(3:end,1);

inputData = imageDatastore(fullfile(Folder,inputFolder),'FileExtensions',...
    '.mat','ReadFcn',@matRead);

all_names = inputData.Files;
for ti = 1:1:size(inputData.Files,1)
test_names = {'a11','a20','a29','b10','b17'};
test_test = [4 12 19 34 41];
% test_test = [10 17 28 37 46];
valid_names = {'a06','a13','a23','b11','b08'};
conv_names = {'a06','a07','a10','a11','a12','a13','a14','a15','a16','a17','a19','a20','a21',...
    'a22','a24','a25','a26','a27','a29','a30','a31','a32','a33','a34',...
    'b-6','b-5','b-4','b-3','b-2','b-1','b07','b08','b09','b10',...
    'b11','b12','b13','b14','b15','b16','b17','b18','b19','b20',...
    'b21','b22','b23','b24'};
    theName = all_names{ti};
    for tc = 1:1:size(test_names,2)
        if contains(theName,test_names{tc})
            test_idx(ti) = logical(1);
            break;
        else
            test_idx(ti) = logical(0);
        end
    end
    
    for tc = 1:1:size(valid_names,2)
        if contains(theName,valid_names{tc})
            valid_idx(ti) = logical(1);
            break;
        else
            valid_idx(ti) = logical(0);
        end
    end
end
train_idx = ~test_idx & ~valid_idx;


% load a specifiy network folder
S = dir([Folder,saveFold]);
S(1:2,:) = [];
S = S(~[S.isdir]);
[~,idx] = sort(([S.datenum]));
S = S(idx);
% maxNum = size(S,1);
% loop over the epochs
% maxNum = 50;
Val_diff = zeros(maxNum,5);
Val_mae = zeros(maxNum,5);
Val_mse = zeros(maxNum,5);
Val_dce = zeros(maxNum,5);
Val_det = zeros(maxNum,5);
Val_tme = zeros(sum(test_idx));


predFold = 'PredictionsFinal';
if isOnlyTrain
    numTestPat = size(test_names,2);
else
    numTestPat = size(conv_names,2);
end

allVals = zeros(5,numTestPat,5);

Val_Alldce = zeros(maxNum,numTestPat);
Val_Alldet = zeros(maxNum,numTestPat);

allT1 = zeros(240,240,60,numTestPat);
allpT1 = zeros(240,240,60,numTestPat);
allT2star = zeros(240,240,60,numTestPat);
allpT2star = zeros(240,240,60,numTestPat);
allWM = zeros(240,240,60,numTestPat);
allpWM = zeros(240,240,60,numTestPat);
allGM = zeros(240,240,60,numTestPat);
allpGM = zeros(240,240,60,numTestPat);
allLes = zeros(240,240,60,numTestPat);
allpLes = zeros(240,240,60,numTestPat);
clear allDiff allDice;
allDiff = zeros(maxNum,2);
allDice = zeros(maxNum,3);

for num=minNum:1:maxNum
Count = 1;
mkdir([Folder,predFold,'/Predictions_Epoch',num2str(num),'_',losses,add2,'_',outputFolder]);
    
newStr = strtrim(S(num,:).name);
tmpStr = strsplit(newStr,'__');
load([Folder,saveFold,'\',newStr]);
tic;
oldTestPatNum = 0;
oldslcNum = 0;
myPatCount = 1;
for folderNum = 1:1:size(allFiles,1)        % loop over all slices
	Fnames = strtrim(allFiles(folderNum,:).name);   
    if (isOnlyTrain && ~test_idx(:,folderNum)) || ~contains(Fnames,'_00.mat')
        continue;      
    end
    
    if isOnlyTrain
        testPatNum = find(contains(test_names,Fnames(1:3)));
    else
        testPatNum = find(contains(conv_names,Fnames(1:3)));
    end
    
    if oldTestPatNum~=testPatNum
        Count = 1;
        oldTestPatNum=testPatNum;
    else
        oldTestPatNum=testPatNum;
    end
    
    slcNum = str2double(Fnames(5:6));
    if (slcNum<oldslcNum)
        myPatCount = myPatCount+1;
%         Fnames
    end
	oldslcNum = slcNum;
        
        
%     im_pred = load([Folder,inputFolder,'\',Fnames]).ttmp;

    load([Folder,outputFolder,'\',Fnames],'tmp')
    im_true = tmp;
    
    try 
        load([Folder,predFold,'/Predictions_Epoch',num2str(num),'_',losses,add2,'_',outputFolder,'\',Fnames],'im_pred')
    catch
        im_pred = readimage(inputData,folderNum);
        if isPatches
            im_pred = predictPatches(net,im_pred);
        else
            im_pred = predict(net, im_pred);
        end
        save([Folder,predFold,'/Predictions_Epoch',num2str(num),'_',losses,add2,'_',outputFolder,'\',Fnames],'im_pred');
    end
    im_true(im_true>300) = 300;
    im_pred(im_pred>300) = 300;
    im_pred(im_pred<0) = 0;
    
    if size(im_pred,3) == 1
    	allpLes(:,:,slcNum,myPatCount) = im_pred(:,:);
        allLes(:,:,slcNum,myPatCount) = im_true(:,:);
    else
        allpT1(:,:,slcNum,myPatCount) = im_pred(:,:,1);
        allpT2star(:,:,slcNum,myPatCount) = im_pred(:,:,2);
        allpWM(:,:,slcNum,myPatCount) = im_pred(:,:,3);
        allpGM(:,:,slcNum,myPatCount) = im_pred(:,:,4);
        allpLes(:,:,slcNum,myPatCount) = im_pred(:,:,5);

        allT1(:,:,slcNum,myPatCount) = im_true(:,:,1);
        allT2star(:,:,slcNum,myPatCount) = im_true(:,:,2);
        allWM(:,:,slcNum,myPatCount) = im_true(:,:,3);
        allGM(:,:,slcNum,myPatCount) = im_true(:,:,4);
        allLes(:,:,slcNum,myPatCount) = im_true(:,:,5);
    end
    Count = Count+1;
        
end


    if size(im_pred,3) == 1
        for tpn = 1:1:numTestPat
            allVals(:,tpn,5) = calculateMetricies(allLes(:,:,:,tpn),allpLes(:,:,:,tpn),'Threshold',85,'Detect');
        end
            Val_Alldce(num,:) = squeeze(meanzeros(allVals(4,:,5),1));
            Val_Alldet(num,:) = squeeze(meanzeros(allVals(5,:,5),1));
        
    else
        for tpn = 1:1:numTestPat
            allVals(:,tpn,1) = calculateMetricies(allT1(:,:,:,tpn),allpT1(:,:,:,tpn));
            allVals(:,tpn,2) = calculateMetricies(allT2star(:,:,:,tpn),allpT2star(:,:,:,tpn));
            allVals(:,tpn,3) = calculateMetricies(allWM(:,:,:,tpn),allpWM(:,:,:,tpn),'Threshold',204);
            allVals(:,tpn,4) = calculateMetricies(allGM(:,:,:,tpn),allpGM(:,:,:,tpn),'Threshold',204);
            allVals(:,tpn,5) = calculateMetricies(allLes(:,:,:,tpn),allpLes(:,:,:,tpn),'Threshold',85,'Detect');
        end

        Val_diff(num,:) = squeeze(meanzeros(allVals(1,:,:),2));
        Val_mae(num,:) = squeeze(meanzeros(allVals(2,:,:),2));
        Val_mse(num,:) = squeeze(meanzeros(allVals(3,:,:),2));
        Val_dce(num,:) = squeeze(meanzeros(allVals(4,:,:),2));
        Val_det(num,:) = squeeze(meanzeros(allVals(5,:,5),2));

        Val_Alldce(num,:) = squeeze(meanzeros(allVals(4,:,5),1));
        Val_Alldet(num,:) = squeeze(meanzeros(allVals(5,:,5),1));
        
    end
fprintf('%s: Epoch:%d finished after %0.2f secs \n', datestr(now),num,toc);

    if 1==1
        slc = 1:60;ixx=test_test;
        if isOnlyTrain
            ixx=1:5;
        end
        tt = abs(allpT1(:,:,slc,ixx)-allT1(:,:,slc,ixx))./allT1(:,:,slc,ixx);
        theWM = allpWM(:,:,slc,ixx);
        theGM = allpGM(:,:,slc,ixx);
        theLes = allLes(:,:,slc,ixx);
        tt(theGM<200 & theWM<200 & theLes<85) = 0;
        tt(isinf(tt))=0;
        allDiff(num,1) = meanzeros(tt(:));

        tt = abs(allpT2star(:,:,slc,ixx)-allT2star(:,:,slc,ixx))./allT2star(:,:,slc,ixx);
        tt(theGM<200 & theWM<200 & theLes<85) = 0;
        tt(isinf(tt))=0;
        allDiff(num,2) = meanzeros(tt(:));

        allDice(num,1)=dice(im2bw(allpWM(:,:,slc,ixx),0.8),im2bw(allWM(:,:,slc,ixx),0.8));
        allDice(num,2)=dice(im2bw(allpGM(:,:,slc,ixx),0.8),im2bw(allGM(:,:,slc,ixx),0.8));
        tt = allpLes(:,:,slc,ixx);tt(tt<85) = 0;tt(tt>0) = 1;
        allDice(num,3) = dice(logical(tt),im2bw(allLes(:,:,slc,ixx),0.8));
    end

end
if minNum~=maxNum
    save(['Epochs_',losses,add2,'-',num2str(size(im_pred,3)),'_',num2str(num),add],'Val_Alldce','Val_Alldet',...
        'Val_dce','Val_det','Val_diff','Val_mae','Val_mse','Val_tme','allDice','allDiff');
    
    figure;
    subplot(1,2,1);
    hold on;
    plot(meanzeros(Val_Alldce(:,:),2))
    plot(allDice(:,:))
    subplot(1,2,2);
    hold on;
    plot(Val_diff)
    plot(allDiff(:,1))
    plot(allDiff(:,2))
    

else

    tic;
    NumberOfThresholds = 50;
    clear myVals myVals2;
    for j=1:1:NumberOfThresholds
        thresh = (j-1)*255/(NumberOfThresholds-1);
        for tpn = 1:1:numTestPat
        [myVals(:,tpn,j) myVals2(:,tpn,j)] = calculateMetricies(allLes(:,:,:,tpn),allpLes(:,:,:,tpn),'Threshold',thresh,'Detect');
        end
        fprintf('%s: Threshold:%d finished after %0.2f secs \n', datestr(now),thresh,toc);
    end
    save(['Thresh_',losses,add2,'-',num2str(size(im_pred,3)),'_',num2str(num),add],'myVals','myVals2');

    figure;
    subplot(1,2,1);hold on;
    plot(squeeze(meanzeros(myVals(4,:,:),1))','Color',myColorMap('Matlab','num',1))
    plot(squeeze(meanzeros(myVals(4,test_test,:),1))','Color',myColorMap('Matlab','num',2))
    plot(squeeze(meanzeros(myVals(4,test_test,:),2))','k-','LineWidth',2)
    subplot(1,2,2);hold on;
    plot(squeeze(meanzeros(myVals(5,:,:),1))','Color',myColorMap('Matlab','num',1))
    plot(squeeze(meanzeros(myVals(5,test_test,:),1))','Color',myColorMap('Matlab','num',2))
    plot(squeeze(meanzeros(myVals(5,test_test,:),2))','k-','LineWidth',2)

end

%% calculate Dice for different Thresholds
tic
tresh_dce = zeros(255,1);
tresh_WM = zeros(255,1);
tresh_GM = zeros(255,1);
tresh_tme = zeros(255,1);
for tresh=1:25:255
    tmp = allLes(:,:,:,:);tmpp = allpLes(:,:,:,:);
    tmp(tmp<26) = 0;tmp(tmp>0) = 1;tmpp(tmpp<tresh) = 0;tmpp(tmpp>0) = 1;
    tresh_dce(tresh) = dice(tmp(:),tmpp(:));
    
    tmp = allWM;tmpp = allpWM;
    tmp(tmp<204) = 0;tmp(tmp>0) = 1;tmpp(tmpp<tresh) = 0;tmpp(tmpp>0) = 1;
    tresh_WM(tresh) = dice(tmp(:),tmpp(:));
    
    tmp = allGM;tmpp = allpGM;
    tmp(tmp<204) = 0;tmp(tmp>0) = 1;tmpp(tmpp<tresh) = 0;tmpp(tmpp>0) = 1;
    tresh_GM(tresh) = dice(tmp(:),tmpp(:));
    
    tresh_tme(tresh) = tresh;
end
tresh_dce(tresh_tme==0) = [];
tresh_WM(tresh_tme==0) = [];
tresh_GM(tresh_tme==0) = [];
tresh_tme(tresh_tme==0) = [];

figure;hold on;
plot(tresh_tme,tresh_dce,'Linewidth',1.5);
plot(tresh_tme,tresh_WM,'Linewidth',1.5);
plot(tresh_tme,tresh_GM,'Linewidth',1.5);
xlabel('Threshold for binarize');
ylabel('Dice between predicted and true');
toc;

Ingo_Beautify_Plot();
%% ploz the MAE, Difference, Dice and Lesion detection rate for the different outputs
Val_tme = 1:maxNum;
Val_tme(Val_mae(:,1)==0) = [];
clrs = get(gca,'colororder');
clrs(5,:) = clrs(3,:);
clrs(3,:) = [0.6 0.6 0.6];
clrs(4,:) = [0.3 0.3 0.3];

sign = '-';
% figure;
subplot(2,2,1);hold on;
for i=1:1:5
    h(i) = plot(Val_tme,squeeze(Val_mae(Val_tme,i)),sign,'Color',clrs(i,:),'Linewidth',1.5);
end
ylabel('RMSE');
legend('T_1','T_2*','WM','GM','Lesions');

subplot(2,2,2);hold on;
for i=1:1:2
    plot(Val_tme,squeeze(Val_diff(Val_tme,i)),sign,'Color',clrs(i,:),'Linewidth',1.5);
end
ylabel('relative difference');

subplot(2,2,3);hold on;
for i=3:1:5
    plot(Val_tme,squeeze(Val_dce(Val_tme,i)),sign,'Color',clrs(i,:),'Linewidth',1.5);
end
ylabel('Dice coefficient');

subplot(2,2,4);hold on;
plot(Val_tme,squeeze(Val_det(Val_tme,i)),sign,'Color',clrs(3,:),'Linewidth',1.5);
ylabel('Detection rate');

Ingo_Beautify_Plot();


%% Predict one image
tic;
num = 160;
mP = 40;
% isPatches = true;
im_true = readimage(outputData,num);

im_pred = readimage(inputData,num);
if isPatches
	im_pred = predictPatches(net, im_pred);
else
	im_pred = predict(net, im_pred);
end


t = toc;

fprintf(['Time needed for predicting one slice: ',num2str(round(t,2)),'s \n']);


n=3;
figure;
subplot(2+n,3,1);imagesc(squeeze(im_true(:,:,1)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,2);imagesc(squeeze(im_pred(:,:,1)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,3);imagesc(100.*squeeze(abs(im_pred(:,:,1)-im_true(:,:,1))./im_true(:,:,1)),[0 mP]);
axis off;axis image;colorbar;

subplot(2+n,3,4);imagesc(squeeze(im_true(:,:,2)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,5);imagesc(squeeze(im_pred(:,:,2)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,6);imagesc(100.*squeeze(abs(im_pred(:,:,2)-im_true(:,:,2))./im_true(:,:,2)),[0 mP]);
axis off;axis image;colorbar;

% figure;
if n>0
subplot(2+n,3,1+6);imagesc(squeeze(im_true(:,:,3)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,2+6);imagesc(squeeze(im_pred(:,:,3)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,3+6);imagesc(100.*squeeze(abs(im_pred(:,:,3)-im_true(:,:,3))./im_true(:,:,3)),[0 mP]);
axis off;axis image;colorbar;

subplot(2+n,3,4+6);imagesc(squeeze(im_true(:,:,4)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,5+6);imagesc(squeeze(im_pred(:,:,4)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,6+6);imagesc(100.*squeeze(abs(im_pred(:,:,4)-im_true(:,:,4))./im_true(:,:,4)),[0 mP]);
axis off;axis image;colorbar;


subplot(2+n,3,7+6);imagesc(squeeze(im_true(:,:,5)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,8+6);imagesc(squeeze(im_pred(:,:,5)),[0 255]);
axis off;axis image;colorbar;
subplot(2+n,3,9+6);imagesc(100.*squeeze(abs(im_pred(:,:,5)-im_true(:,:,5))./im_true(:,:,5)),[0 50]);
axis off;axis image;colorbar;
end
