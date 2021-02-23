function [output] = predictHiddenPatches(net,input,layerNumber,varargin)

if max(strcmp(varargin,'patchSize'))
    idx = 1 + find(strcmp(varargin,'patchSize'));
    dim = varargin{1,idx};
else
    dim = 64;
end

if max(strcmp(varargin,'is3D'))
    fprintf('Not working yet!!!');
    if max(strcmp(varargin,'slices'))
        idx = 1 + find(strcmp(varargin,'slices'));
        slc = varargin{1,idx};
    else
        slc = 16;
    end

    out = predict(net, input(1:dim,1:dim,1:slc,:) );
    if max(strcmp(varargin,'dim'))
        idx = 1 + find(strcmp(varargin,'dim'));
        outDim = varargin{1,idx};
    else
        outDim = [240 240 60 size(out,4)];
    end
    dimFac = size(input,1)-dim;
    dimFacSlc = size(input,3)-slc+1;

    cc = 1;offset=0;dimStep = 47;dimStepSlc = 14;offsetSlc=0;
    dimLen = ceil(dimFac/dimStep)+ceil(dimFacSlc/dimStepSlc);

    tmn = zeros(outDim(1),outDim(2),outDim(3),outDim(4),dimLen^2+1);
    for ci=1:dimStep:dimFac
        for cj=1:dimStep:dimFac
            for ck=1:dimStepSlc:dimFacSlc
%             parfor cc=1:ceil(dimFacSlc/dimStepSlc)
%                 ck = 1+(cc-1)*dimStepSlc;
             
                x = [0:dim-1]+ci;
                y = [0:dim-1]+cj;
                z = [0:slc-1]+ck;
                tmp = input(x,y,z,:);
                tmp2 = predict(net, tmp);

%                 tmn(:,:,:,:,cc) = ...
%                     tmp2(offset:end-offset,offset:end-offset,offsetSlc:end-offsetSlc,:);
                tmn(x(1+offset:end-offset),y(1+offset:end-offset),z(1+offsetSlc:end-offsetSlc),:,cc) = ...
                    tmp2(1+offset:end-offset,1+offset:end-offset,1+offsetSlc:end-offsetSlc,:);
                cc = cc+1;
            end

        end
    end
    %         tmp3 = median(tmn,4);
    tmp3 = meanzeros(tmn,5);
    im_pred = tmp3;

    output = im_pred;

    
else
    
    theLayers = net.Layers;
    theLayer = theLayers(layerNumber).Name;
    fprintf('You chose the layer: %s\n',theLayer);
%     out = predict(net, input(1:dim,1:dim,:) );
    out = activations(net,input(1:dim,1:dim,:) ,theLayer);
    if max(strcmp(varargin,'dim'))
        idx = 1 + find(strcmp(varargin,'dim'));
        outDim = varargin{1,idx};
    else
        outDim = [240 240 size(out,3)];
    end
    dimFac = size(input,1)-dim;
    stretchFac = dim/size(out,1);

    cc = 1;offset=10;dimStep = 24;dimLen = ceil(dimFac/dimStep);

    tmn = zeros(outDim(1),outDim(2),outDim(3),dimLen^2+1);
    for ci=1:dimStep:dimFac
        for cj=1:dimStep:dimFac
            x = [1:dim]+ci;
            y = [1:dim]+cj;
            tmp = input(x,y,:);
%             tmp2 = predict(net, tmp);
            tmp2 = activations(net,tmp ,theLayer);
            tmp2 = imresize(tmp2,[dim dim]);
            tmn(x(offset:end-offset),y(offset:end-offset),:,cc) = tmp2(offset:end-offset,offset:end-offset,:);
            
%             nx = x/2;
%             ny = y/2;
%             noffset = offset/2;
%             tmn(nx(noffset:end-noffset),ny(noffset:end-noffset),:,cc) = tmp2(noffset:end-noffset,noffset:end-noffset,:);
            cc = cc+1;

        end
    end
    %         tmp3 = median(tmn,4);
    tmp3 = meanzeros(tmn,4);
    im_pred = tmp3;

    output = im_pred;
end
