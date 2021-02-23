% PredictOnePatient

Folder = 'F:\users\ih11\DeepLearning MRFMS\MRFReco_Full\';
inputFolder = 'RotMRF';
outputFolder = 'RotT1T2starMaskLesions';

saveFold = 'aCheckFolder_gpu_mse_patches_RotT1T2starMaskLesions1000';
predFolder = [Folder,'Predictions\Predictions_Epoch100_mse_',outputFolder];

allFiles = dir([Folder,inputFolder]);
allFiles = allFiles(3:end,1);

S = dir([Folder,saveFold]);
S(1:2,:) = [];
S = S(~[S.isdir]);
[~,idx] = sort(([S.datenum]));
S = S(idx);

% loop over the epochs
maxNum = size(S,1);
newStr = strtrim(S(maxNum,:).name);
tmpStr = strsplit(newStr,'__');
load([Folder,saveFold,'\',newStr]);

test_names = {'a11','a20','a29','b10','b17'};
valid_names = {'a06','a13','a23','b11','b08'};

c = 1;
allNames(1,:) = '000';
for i=1:1:size(allFiles,1)
    Fnames = strtrim(allFiles(i,:).name);   
    tmpName = Fnames(1:3);
    if max(sum(ismember(allNames,tmpName,'rows'),2))
        continue;
    else
        allNames(c,:) = tmpName;
        c = c+1;
    end
end
for theC = 1:1:size(allNames,1)
    theName = allNames(theC,:);
    if contains(theName,test_names)
        test_pats(theC) = 1;
    else
        test_pats(theC) = 0;
    end
    if contains(theName,valid_names)
        val_pats(theC) = 1;
    else
        val_pats(theC) = 0;
    end
allT1 = zeros(240,240,60);
allpT1 = zeros(240,240,60);
allT2star = zeros(240,240,60);
allpT2star = zeros(240,240,60);
allWM = zeros(240,240,60);
allpWM = zeros(240,240,60);
allGM = zeros(240,240,60);
allpGM = zeros(240,240,60);
allLes = zeros(240,240,60);
allpLes = zeros(240,240,60);

Count = 1;
    
tic;
for folderNum = 1:1:size(allFiles,1)        % loop over all slices
	Fnames = strtrim(allFiles(folderNum,:).name);   
    if contains(Fnames,theName) && contains(Fnames,'_00.mat')  
        if Count == 1
            fprintf('%s: Folder: %s\n',datestr(now),Fnames(1:end-4));
        end
    else
        continue;      
    end
    
    slcNum = str2double(Fnames(5:6));
        
    load([Folder,outputFolder,'\',Fnames],'tmp')
    im_true = tmp;
    
    
    try
        load([predFolder,'\',Fnames]);
    catch
        load([Folder,inputFolder,'\',Fnames],'ttmp')
        im_pred = readimage(inputData,folderNum);
        if isPatches
            im_pred = predictPatches(net, ttmp,'patchSize',64);
        else
            im_pred = predict(net, im_pred);
        end
        save([predFolder,'\',Fnames],'im_pred');
    end
        
        
    im_true(im_true>300) = 300;
    im_pred(im_pred>300) = 300;
    im_pred(im_pred<0) = 0;
    
    allpT1(:,:,slcNum) = im_pred(:,:,1);
    allpT2star(:,:,slcNum) = im_pred(:,:,2);
    allpWM(:,:,slcNum) = im_pred(:,:,3);
    allpGM(:,:,slcNum) = im_pred(:,:,4);
    allpLes(:,:,slcNum) = im_pred(:,:,5);

    allT1(:,:,slcNum) = im_true(:,:,1);
    allT2star(:,:,slcNum) = im_true(:,:,2);
    allWM(:,:,slcNum) = im_true(:,:,3);
    allGM(:,:,slcNum) = im_true(:,:,4);
    allLes(:,:,slcNum) = im_true(:,:,5);

    Count = Count+1;
        
end

parameters(:,1) = calculateMetricies(allT1,allpT1);
parameters(:,2) = calculateMetricies(allT2star,allpT2star);
parameters(:,3) = calculateMetricies(allWM,allpWM,'Treshold',204);
parameters(:,4) = calculateMetricies(allGM,allpGM,'Treshold',204);
parameters(:,5) = calculateMetricies(allLes,allpLes,'Treshold',26,'Detect');

time(theC) = toc;
% allDice(theC) = parameters(4,5);
% allDet(theC) = parameters(5,5);

allDiff(theC,:) = parameters(1,:);
allMAE(theC,:) = parameters(2,:);
allMSE(theC,:) = parameters(3,:);
allDice(theC,:) = parameters(4,:);
allDet(theC,:) = parameters(5,:);
end

% parameters
pats = 1:theC;
figure;
subplot(2,2,1);hold on;
plot(pats,allDiff(:,1:4),'.');

subplot(2,2,2);hold on;
plot(pats,allMSE(:,1:4),'.');

subplot(2,2,3);hold on;
plot(pats,allDet(:,5),'.');
plot(pats(test_pats==1),allDet(test_pats==1,5),'.');

subplot(2,2,4);hold on;
plot(pats,allDice(:,3:5),'.');
plot(pats(test_pats==1),allDice(test_pats==1,3:5),'.');

