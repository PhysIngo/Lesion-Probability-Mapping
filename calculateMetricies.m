function [output,varargout] = calculateMetricies(im_true,im_pred,varargin)
% calculateMetricies by Ingo Hermann, 2020-11-25
% This functions calculate measures as the MAE, Dice and so on
% --------------------------------
% This script needs the user functions:
% - none -
%
% Exp:. [output,varargout] = calculateMetricies(im_true,im_pred,varargin)
%
% simulatePhaseBasedT2(varargin):
% 'Threshold',thresh ... sets the threshold for binarizing into masks
% 'Detect' ... additionally calculates the lesion detection rate


im_true(im_true<1) = 0;
im_pred(im_pred<1) = 0;

im_true(im_true>255) = 255;
im_pred(im_pred>255) = 255;

im_diff = abs(im_pred-im_true)./im_true;
Val_diff = mean(im_diff(im_true~=0),1);

im_mae = abs(im_pred-im_true);
im_mae(im_true==0) = 0;
Val_mae = mean(im_mae(im_true~=0),1);

im_mse = abs(im_pred-im_true).^2;
im_mse(im_true==0) = 0;
Val_mse = (mean(im_mse(im_true~=0),1)).^0.5;

%binarize
if max(strcmp(varargin,'Threshold'))
    idx = 1 + find(strcmp(varargin,'Threshold'));
    thresh = varargin{1,idx};
    im_pred(im_pred<thresh) = 0;
    im_pred(im_pred>0) = 1;
    im_true(im_true<thresh) = 0;
    im_true(im_true>0) = 1;

    % calculate the Dice here
    common = (im_true & im_pred);
    a = sum(common(:));
    b = sum(im_true(:));
    c = sum(im_pred(:));
    Dice = 2*a/(b+c);
    if Dice<2
        Val_dce = Dice;
    else
        Val_dce = 0;
    end    

    if max(strcmp(varargin,'Detect'))
        % calculate the Lesion detection rate here
        tmpLes = SeparateLesions(im_true);
        numLes = unique(tmpLes(:));
        lesDetection = 0;
        for lc=1:1:max(numLes)
            tmpMask = tmpLes;tmpMask(tmpMask~=lc) = 0;
            if meanzeros(im_pred(tmpMask~=0),1) ~= 0
                lesDetection = lesDetection+1;
            end
        end
        tt = (lesDetection)/(length(numLes)-1);tt(isnan(tt)) = 0;
        Val_det = tt;
    else
        Val_det = 0;
    end
    
    tmp = im_pred(:).*2+im_true(:);
    FN = sum(tmp==0)./length(im_pred(:));
    FP = sum(tmp==1)./length(im_pred(:));
    TN = sum(tmp==2)./length(im_pred(:));
    TP = sum(tmp==3)./length(im_pred(:));
%     end
    Val_acc = (TP+TN)/(TP+TN+FP+FN);
    Val_spe = (TN)/(TN+FP);
    Val_sen = (TP)/(TP+FN);
    Val_dic = (2*TP)/(2*TP+FP+FN);
    varargout{1} = [TP, FP, TN, FN, Val_acc, Val_spe, Val_sen, Val_dic];
else
    Val_dce = 0;
    Val_det = 0;
    Val_TP = 0;
    Val_FP = 0;
    Val_TN = 0;
    Val_FN = 0;
    Val_acc = 0;
    varargout{1} = [];
end

% output
output = [Val_diff, Val_mae, Val_mse, Val_dce, Val_det];
        

end