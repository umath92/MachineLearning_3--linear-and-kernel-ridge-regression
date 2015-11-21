function Q5()


g=[1,2,3,4,5,6];
g=g';
% samplesize=10
% part (a)
[var,biasSquare]=heart(10);
fprintf(sprintf( '\n Variance for 10 sample: \n'));
table(var,biasSquare,g)

%figure;
%plot(g,var,g,biasSquare)


% samplesize=100
% part(b)
[var2,biasSquare2]=heart(100);
fprintf(sprintf( '\n Variance for 100 sample: \n'));
table(var2,biasSquare2,g)



%part d
lambda=[0.01, 0.1, 1, 10]';
[variance,bias,error]=partD(100);
error=error';
figure;
subplot(2,1,1)
plot(log10(lambda),bias,log10(lambda),variance)
title('Part d')
xlabel('log10(lambda)');
legend('bias','variance');

subplot(2,1,2)
set(gca,'ytick',linspace(-1,6,10))
plot(log10(lambda),error)



xlabel('log10(lambda)');
ylabel('error');

fprintf(sprintf( '\n Part d: \n'));

table(lambda,variance,bias)


end

function [variance,biass]=heart(samplesize)
    % doing it for 10 points and g1
    % y = w0 +w1x+w2x2

    index=[-1,0,1,2,3,4];
    figure;
    iterationsCount=100;
    varIndex=1;
    biasIndex=1;
    va=1;
    
    averagePredict=zeros(samplesize,iterationsCount);
    
    inputB=(-1+rand(1,samplesize)*(2))';
    
    for i=1:size(inputB,1)
        Y_T(i)=2*inputB(i)*inputB(i);
    end
    
    for k=1:size(index,2)
        error=zeros(iterationsCount,1);
        for interation=1:iterationsCount
            input=(-1+rand(1,samplesize)*(2))';
            Y=getY(input)';
            
            
            if(index(k)==-1)
                for row=1:size(Y,1)
                    error(interation)=error(interation)+power((Y(row)-1),2);
                end
                error(interation)=error(interation)/size(Y,1);
                var(varIndex)=mean(power((1-1),2));
                varIndex=varIndex+1;
                
                bias(biasIndex)=power((1-mean(Y)),2);
                biasIndex=biasIndex+1;
            else
                X=[];
                for i=1:index(k)
                    X=horzcat(X,power(input,i));
                end
                X=horzcat(ones(size(Y,1),1),X);
                theta=solution(X,Y);
                % get sum-square error
                predict=X*theta;
                
                
                %%%Get true Y
            
                Y_true=2*power(X,2);
            
                %%%Get true Y
                
                for row=1:size(Y,1)
                    error(interation)=error(interation)+power((Y(row)-predict(row)),2);
                end
                error(interation)=error(interation)/size(predict,1);
                var(varIndex)=mean(power((predict-mean(predict)),2));
                varIndex=varIndex+1;
                
                bias(biasIndex)=power((mean(predict)-mean(Y)),2);
                biasIndex=biasIndex+1;
                
                
               
            end
            % New way to calculate bias
            X=[];
            for i=1:index(k)
                X=horzcat(X,power(inputB,i));
            end

            Y=getY(inputB)';

            X=horzcat(ones(size(Y,1),1),X);
            theta=solution(X,Y);
            % get sum-square error
            predict=X*theta;
            averagePredict(:,va)=predict;
            va=va+1;
            %%%%%%%%%%%%%
            
            
        end
        subplot(3,2,k);
        %error=error.*(1/iterationsCount);
        hist(error,10)
        title( char( sprintf( 'Function g=g%d', k ) ) );
        ylabel('MSE');
        xlabel('Bins');
        
        %Variance Estimation
        varIndex=1;
        variance(k)=mean(var);
        
        %Bias Estimation
        biasIndex=1;
        biass(k)=mean(bias);
        
        %%%%%%%%%%NEW WAY%%%%%%%%%%%%%%%%%%%%%%%%
        %averagePredict
        bias=0;
        for i=1:size(Y_T,2)
            bias=bias+power(Y_T(i)-mean(averagePredict(i,:)),2);
            var=power(averagePredict(i,1)-mean(averagePredict(i,:)),2);
        end
        bias=bias/size(Y_T,2);
        biass(k)=bias;
        if(k==0)
            variance(k)=0;
        elseif (k==1)
            variance(k)=0;
        else
            variance(k)=var/size(Y_T,2);
        end
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    variance=variance';
    biass=biass';
end


function [variance,biass,errori]=partD(samplesize)
    % doing it for 10 points and g1
    % y = w0 +w1x+w2x2

    index=[2];
    iterationsCount=100;
    lambda=[0.01, 0.1, 1, 10];
    varIndex=1;
    biasIndex=1;
    
    for l=1:size(lambda,2)
            error=zeros(iterationsCount,1);
            for interation=1:iterationsCount
                input=(-1+rand(1,samplesize)*(2))';
                Y=getY(input)';
                X=[];
                for i=1:2
                    X=horzcat(X,power(input,i));
                end
                X=horzcat(ones(size(Y,1),1),X);
                theta=solutionLambda(X,Y,lambda(l));
                % get sum-square error
                predict=X*theta;
                %%%Get true Y
            
                Y_true=2*power(X,2);
                for g=1:samplesize
                    Y_bias_true(interation,g)=2*input(g)*input(g);
                end
            
                %%%Get true Y
                for row=1:size(Y,1)
                    error(interation)=error(interation)+power((Y(row)-predict(row)),2);
                end
                
                error(interation)=error(interation)/size(predict,1);
                var(varIndex)=mean(power((predict-mean(predict)),2));
                varIndex=varIndex+1;
                Y_bias_exp(interation,:)=predict';
                bias(biasIndex)=power((mean(predict)-mean(Y)),2);
                biasIndex=biasIndex+1;
            end
            errori(l)=mean(error);
            %hist(error.*(1/iterationsCount),10)
            %title( char( sprintf( 'Function g=g%d', k ) ) );
            %ylabel('MSE');
            %xlabel('Bins');

            %Variance Estimation
            varIndex=1;
            variance(l)=mean(var);

            %Bias Estimation
            biasIndex=1;
            biass(l)=mean(bias);
    end
    
    variance=variance';
    biass=biass';
end



function [theta]= solutionLambda(X,Y,lambda)
    theta=(transpose(X)*X + lambda*eye(size(X,2),size(X,2)))\(transpose(X)*Y);
end



function [theta]= solution(X,Y)
    theta=(transpose(X)*X)\(transpose(X)*Y);
end

function [Y] = getY(X)
    for row=1:size(X,1)
        Y(row)=2*power(X(row),2)+gaussian(X(row),0,0.1);
    end 
end



function [s]=gaussian(x,u,var)
    s=(1/(sqrt(2*pi*var)))*exp(-power((x-u),2)/(2*var));
end