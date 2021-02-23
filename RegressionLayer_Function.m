classdef RegressionLayer_Function < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = RegressionLayer_Function(name)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = "myRegressionLayer";

            % Set layer description.
            if strcmp(name,'mse')
                layer.Description = 'MSE';
            elseif strcmp(name,'rl')
                layer.Description = 'RL';
            elseif strcmp(name,'mre')
                layer.Description = 'MRE';
            elseif strcmp(name,'mle')
                layer.Description = 'MLE';
            elseif strcmp(name,'huber')
                layer.Description = 'HL';
            elseif strcmp(name,'lcl')
                layer.Description = 'HL';
            elseif strcmp(name,'weight')
                layer.Description = 'WEIGHT';
            elseif strcmp(name,'dice')
                layer.Description = 'DICE';
            elseif strcmp(name,'log')
                layer.Description = 'LOG';
            else
                layer.Description = 'MAE';
            end
                
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            if strcmp(layer.Description,'MAE')
                % Calculate MAE.
                R = size(Y,3);
                meanAbsoluteError = sum(abs(Y-T),3)./R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanAbsoluteError(:))./N;
                
            elseif strcmp(layer.Description,'MSE')
                % Calculate MAE.
                R = size(Y,3);
                meanSquaredError = sum(abs(Y-T).^2,3)./R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanSquaredError(:))./N;
                           
            elseif strcmp(layer.Description,'MLE')
                % Calculate MAE.
                R = size(Y,3);
                C = sqrt((abs(Y-T)));
                meanSquaredError=sum((log((abs(Y)+1)./(abs(T)+1))).^2,3)/R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanSquaredError(:))./N;
                
            elseif strcmp(layer.Description,'MRE')
                % Calculate MAE.
                R = size(Y,3);
                C = sqrt((abs(Y-T)));
%                 meanSquaredError = sum(C,3)./R;
                meanSquaredError=sum(sqrt(abs((abs(Y)+1)./(abs(T)+1)-1)),3)/R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanSquaredError(:))./N;
                            
            elseif strcmp(layer.Description,'HL')
                % Calculate MAE.
                R = size(Y,3);
                D = abs(Y-T);
                delta = 5;
                Y1 = Y;Y1(D<5) = 0;
                Y2 = Y;Y2(D>=5) = 0;
                T1 = T;T1(D<5) = 0;
                T2 = T;T2(D>=5) = 0;
                meanSquaredError = sum( 0.5.*abs(Y2-T2).^2 +  delta.*abs(Y1-T1)-0.5*delta^2 ,3)./R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanSquaredError(:))./N;
                            
            elseif strcmp(layer.Description,'LCL')
                % Calculate MAE.
                R = size(Y,3);
                
                meanSquaredError = sum( log(cosh(Y-T)) ,3)./R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanSquaredError(:))./N;
                
            elseif strcmp(layer.Description,'RL')
                % Calculate MAE.
                R = size(Y,3);
                C = abs(Y-T);
                T2 = T;
                C(T2==0) = 1;
                T2(T==0) = 1;
                
                C2 = abs(Y-T);
                C2(T~=0) = 0;
                %relativeError=sum( abs( ((abs(Y)+1)./(abs(T)+1)-1) ),3)/R;
%                 relativeError = sum( C./T ,3)/R;

%                 relativeError = sum(abs(C./T),3)./R;
                relativeError = sum(100*abs(C./T2)+C2./3,3)./R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(relativeError(:))./N;
                
            elseif strcmp(layer.Description,'DICE')
                % Calculate MAE.
                R = size(Y,3);
                N = size(Y,4);
                
                Y1 = Y;
                T1 = T;
                Y1(Y1<125) = 0;
                Y1(Y1>0) = 1;
                T1(T1<125) = 0;
                T1(T1>0) = 1;
                
%                 tmp = Y1(:).*2+T1(:);
%                 L = sum(size(Y));
%                 tmp1 = tmp;%tmp1(tmp~=0) = [];
%                 FN = sum(tmp1(tmp1==0))./L;
%                 tmp2 = tmp;%tmp2(tmp~=1) = [];
%                 FP = sum(tmp2(tmp2==1))./L;
%                 tmp3 = tmp;%tmp3(tmp~=2) = [];
%                 TN = sum(tmp3(tmp3==2))./L;
%                 tmp4 = tmp;%tmp4(tmp~=3) = [];
%                 TP = sum(tmp4(tmp4==4))./L;
%                 ALL = TP+TN+FP+FN;
%                 dceError = (2*TP)/((FP+TP)*(TP+FN));
                
                
%                 dceError = (TP.^2)/(TP*(TP+FN+TN+FP));
                %dceError = (2*TP/(2*TP+FP));
%                 if dceError>1
%                     dceError = 0;
%                 end

                tmp = Y1(:).*T1(:);
                tmp2 = Y1(:).*Y1(:);
                tmp3 = T1(:).*T1(:);
                dceError = (1+2.*sum(tmp(:)))./(sum(tmp2(:))+sum(tmp3(:))+1);
                
                loss = (1-dceError);
                
                
                
%                 meanSquaredError = sum( log(cosh(Y-T)) ,3)./R;
%                 Take mean over mini-batch.
%                 N = size(Y,4);
%                 loss = sum(meanSquaredError(:))./N;
                
            elseif strcmp(layer.Description,'LOG')
                % Calculate MAE.
                R = size(Y,3);
                N = size(Y,4);
                
                Y1 = Y./255;
                T1 = T./255;
                T1 = (1-Y1+T1)./2;
                T1(T1<0) = 0;
                T1(T1>1) = 1;
                logError = 1-abs(log(abs(T1)));
%                 logError(T1==1) = abs(log(abs(Y1)));
%                 logError(T1==0) = abs(log(1-abs(Y1)));
%                 if logError>1
%                     logError = 0;
%                 end
                loss = 1000*(logError);
                
                
            elseif strcmp(layer.Description,'WEIGHT')
                % Calculate MAE.
                R = size(Y,3);
                Y1 = Y;
                T1 = T;
                Y1(Y1<250) = Y1(Y1<250).*1.5;
                Y1(Y1>=250) = Y1(Y1>=250).*0.5;
                meanSquaredError = sum(abs(Y-T).^2,3)./R;
%                 meanSquaredError = sum(abs(0.8*(Y(T<=250)-T(T<=250))+0.2*(Y(T>250)-T(T>250))).^2,3)./R;
                % Take mean over mini-batch.
                N = size(Y,4);
                loss = sum(meanSquaredError(:))./N;
            end
        end
        
        
%         function dLdY = backwardLoss(layer, Y, T)
            % (Optional) Backward propagate the derivative of the loss 
            % function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     – Predictions made by network
            %         T     – Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the 
            %                 predictions Y        

            % Layer backward loss function goes here.
            
%             [~,~,K,N] = size(Y);
%             W = ones(size(Y));
%             weight = [1, 1, 1, 1];
%             for i=1:1:K
%                 W(:,:,i,:) = W(:,:,i,:).*weight(i);
%             end
%             dLdY = -(W.*T./Y)./N;
             
            
%         end
        
    end
end
