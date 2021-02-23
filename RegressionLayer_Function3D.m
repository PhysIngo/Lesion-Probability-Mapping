classdef RegressionLayer_Function3D < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = RegressionLayer_Function3D(name)
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
            else
                layer.Description = 'MAE';
            end
                
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the MAE loss between
            % the predictions Y and the training targets T.

            if strcmp(layer.Description,'MAE')
                % Calculate MAE.
                R = size(Y,4);
                meanAbsoluteError = sum(abs(Y-T),4)./R;
                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(meanAbsoluteError(:))./N;
                
            elseif strcmp(layer.Description,'MSE')
                % Calculate MAE.
                R = size(Y,4);
                meanSquaredError = sum(abs(Y-T).^2,4)./R;
                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(meanSquaredError(:))./N;
                           
            elseif strcmp(layer.Description,'MLE')
                % Calculate MAE.
                R = size(Y,4);
                C = sqrt((abs(Y-T)));
                meanSquaredError=sum((log((abs(Y)+1)./(abs(T)+1))).^2,4)/R;
                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(meanSquaredError(:))./N;
                
            elseif strcmp(layer.Description,'MRE')
                % Calculate MAE.
                R = size(Y,4);
                C = sqrt((abs(Y-T)));
%                 meanSquaredError = sum(C,3)./R;
                meanSquaredError=sum(sqrt(abs((abs(Y)+1)./(abs(T)+1)-1)),4)/R;
                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(meanSquaredError(:))./N;
                            
            elseif strcmp(layer.Description,'HL')
                % Calculate MAE.
                R = size(Y,4);
                D = abs(Y-T);
                delta = 5;
                Y1 = Y;Y1(D<5) = 0;
                Y2 = Y;Y2(D>=5) = 0;
                T1 = T;T1(D<5) = 0;
                T2 = T;T2(D>=5) = 0;
                meanSquaredError = sum( 0.5.*abs(Y2-T2).^2 +  delta.*abs(Y1-T1)-0.5*delta^2 ,4)./R;
                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(meanSquaredError(:))./N;
                            
            elseif strcmp(layer.Description,'LCL')
                % Calculate MAE.
                R = size(Y,4);
                
                meanSquaredError = sum( log(cosh(Y-T)) ,4)./R;
                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(meanSquaredError(:))./N;
                
            elseif strcmp(layer.Description,'RL')
                % Calculate MAE.
                R = size(Y,4);
                C = abs(Y-T);
                C(T==0) = 0;
                T(T==0) = 1;
                relativeError=sum( abs( ((abs(Y)+1)./(abs(T)+1)-1) ),4)/R;
%                 relativeError = sum( C./T ,3)/R;

                % Take mean over mini-batch.
                N = size(Y,5);
                loss = sum(relativeError(:))./N;
                
            elseif strcmp(layer.Description,'WEIGHT')
                % Calculate MAE.
                R = size(Y,4);
                Y1 = Y;
                T1 = T;
                Y1(Y1<250) = Y1(Y1<250).*1.5;
                Y1(Y1>=250) = Y1(Y1>=250).*0.5;
                meanSquaredError = sum(abs(Y-T).^2,3)./R;
%                 meanSquaredError = sum(abs(0.8*(Y(T<=250)-T(T<=250))+0.2*(Y(T>250)-T(T>250))).^2,3)./R;
                % Take mean over mini-batch.
                N = size(Y,5);
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
