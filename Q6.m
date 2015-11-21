function Q6(part)
    
    %class={'ln(VOTES/POP)', 'POP', 'EDUCATION', 'HOUSES', 'INCOME', 'XCOORD', 'YCOORD'}
    % ln(VOTES/POP) is the dependent variable
    
    data=load('dataset');
    assignin('base', 'data', data);
    
    % Linear Rigid Regression
    
    if (part==1)
        figure;
        for plotNo=1:10
            [train,test]=dataSplit(0.5,data);
            fprintf(sprintf( '\n****** Split#%d ******\n' ,plotNo));
            [avgerror(plotNo)]=linearRigidRegression(train,plotNo);
        end
        fprintf(sprintf( '\n****** Average error ******\n'));
        avgerror=avgerror';
        table(avgerror)
    end
    
    % linear rigid kernel
    if(part==2)
        fprintf(sprintf( '\n********** Linear Rigid Kernel **********\n'));
        figure;
        for plotNo=1:3
            [train,test]=dataSplit(0.5,data);
            fprintf(sprintf( '\n****** Split#%d ******\n' ,plotNo));
            [LinearRigidAverageError(plotNo)]=kernelRigidRegression(train,plotNo,1,test);
        end
        LinearRigidAverageError=LinearRigidAverageError';
        table(LinearRigidAverageError)
    end
    
    % polynomial rigid kernel
    if(part==3)
        fprintf(sprintf( '\n********* Polynomial Rigid Kernel *********\n'));
        for plotNo=1:3
            [train,test]=dataSplit(0.5,data);
            fprintf(sprintf( '\n****** Split#%d ******\n' ,plotNo));
            [PolynomialRigidAverageError(plotNo)]=kernelRigidRegression(train,plotNo,2,test);
            PolynomialRigidAverageError
        end
        PolynomialRigidAverageError=PolynomialRigidAverageError';
        table(PolynomialRigidAverageError)
    end
    
    % Gaussian rigid kernel
    if(part==4)
        fprintf(sprintf( '\n********* Gaussian Rigid Kernel *********\n'));
        for plotNo=1:1
            [train,test]=dataSplit(0.5,data);
            fprintf(sprintf( '\n****** Split#%d ******\n' ,plotNo));
            figure;
            [GaussianRigidAverageError(plotNo)]=kernelRigidRegression(train,plotNo,3,test);
            GaussianRigidAverageError
        end
        GaussianRigidAverageError=GaussianRigidAverageError';
        table(GaussianRigidAverageError)
    end
    
end


function [avgerror]=kernelRigidRegression(train,plotNo,part,test)

    if(part==1)
        % linear rigid kernel
        a=0;
        b=0;
        sigma=1;
        totalSplit=5;
        lambda=[10^-6, 10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3];
        errorLambda=zeros(1,size(lambda,2));
        for l=1:size(lambda,2)
            testingError=0;
            for eachSplit=0:4
                [train1,test1]=getTrainandTest(train,eachSplit,totalSplit);
                [error]=getSplitErrorKernel(train1,test1,lambda(l),1,a,b,sigma);
                testingError=testingError+error;
            end
            errorLambda(l)=testingError/totalSplit;
        end
        
        lambda=lambda';
        errorLambda=errorLambda';
        table(lambda,errorLambda)
        subplot(5,2,plotNo);
        plot(log10(lambda),errorLambda)
        xlabel('log10(lambda)')
        ylabel('error')
        
        [~,I]=min(errorLambda);
        optLambda=lambda(I);
        [error]=getSplitErrorKernel(train,test,optLambda,1,a,b,sigma);
        avgerror=error;
        
    elseif(part==2)
        % polynomial rigid kernal
        
        totalSplit=5;
        lambda=[10^-6, 10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3];
        errorLambda=zeros(1,size(lambda,2));
        A=[-1, -0.5, 0, 0.5, 1];
        B=[1,2,3,4];
        sigma=1;
        i=1;
        tableiter=1;
        for a=1:size(A,2)
            for b=1:size(B,2)
                for l=1:size(lambda,2)
                    testingError=0;
                    for eachSplit=0:4
                        [train1,test1]=getTrainandTest(train,eachSplit,totalSplit);
                        [error]=getSplitErrorKernel(train1,test1,lambda(l),2,A(a),B(b),sigma);
                        testingError=testingError+error;
                    end
                    errorLambda(l)=testingError/totalSplit;
                    errorTable(tableiter)=errorLambda(l);
                    ATable(tableiter)=A(a);
                    BTable(tableiter)=B(b);
                    lambdaTable(tableiter)=lambda(l);
                    tableiter=tableiter+1;
                end
                fprintf(sprintf( '\nValues of a and b: %d %d\n' ,A(a),B(b)));
                lambda=(lambda');
                errorLambda=errorLambda';
                %table(lambda,errorLambda)
                avgerror(i)=(mean(errorLambda));
                %plot(log10(lambda),errorLambda)
                %hold on
                i=i+1;
                lambda=[10^-6, 10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3];
                A=[-1, -0.5, 0, 0.5, 1];
                B=[1,2,3,4];
                errorLambda=zeros(1,size(lambda,2));
            end
        end
        %xlabel('log10(lambda)');
        %ylabel('error');
        %hold off;
        A=ATable';
        B=BTable';
        lambda=lambdaTable';
        error=errorTable';
        table(A,B,lambda,error)
        
        [~,I]=min(error);
        optLambda=lambda(I)
        optA=A(I)
        optB=B(I)
        [error]=getSplitErrorKernel(train,test,optLambda,2,optA,optB,sigma);
        avgerror=error;
        
    
    elseif(part==3)
        % exponential kernel

        totalSplit=5;
        %lambda=[10];
        lambda=[10^-6, 10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3];
        
        errorLambda=zeros(1,size(lambda,2));
        A=1;
        B=1;
        sigma=[0.125, 0.25, 0.5, 1, 2, 4, 8];

        %sigma=[1];
        i=1;
        tableiter=1;
        for a=1:size(sigma,2)
            for l=1:size(lambda,2)
                testingError=0;
                for eachSplit=0:4
                    [train1,test1]=getTrainandTest(train,eachSplit,totalSplit);
                    [error]=getSplitErrorKernel(train1,test1,lambda(l),3,A,B,sigma(a));
                    testingError=testingError+error;
                end
                errorLambda(l)=testingError/totalSplit;
                
                errorTable(tableiter)=errorLambda(l);
                sigmaTable(tableiter)=sigma(a);
                lambdaTable(tableiter)=lambda(l);
                tableiter=tableiter+1;
            end
            %fprintf(sprintf( '\nValue of sigma: %d \n' ,sigma(a)));
            lambda=lambda';
            errorLambda=errorLambda';
            %table(lambda,errorLambda)
            avgerror(i)=(mean(errorLambda));
            plot(log10(lambda),errorLambda)
            hold on
            i=i+1;
            lambda=[10^-6, 10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3];
            sigma=[0.125, 0.25, 0.5, 1, 2, 4, 8];
            errorLambda=zeros(1,size(lambda,2));
        end
        xlabel('log10(lambda)');
        ylabel('error');
        legend('0.125','0.25','0.5','1','2','4','8')
        hold off;
        sigma=sigmaTable';
        lambda=lambdaTable';
        error=errorTable';
        table(sigma,lambda,error)
        
        [~,I]=min(error);
        optSigma=sigma(I)
        optLambda=lambda(I)
        [error]=getSplitErrorKernel(train,test,optLambda,3,A,B,optSigma);
        avgerror=error;
        
    end
    
end


function [error]=getSplitErrorKernel(train1,test1,lambda,which,a,b,sigma)
    X=ones(size(train1,1),1);
    for col=2:size(train1,2)
        X=horzcat(X,train1(:,col));
    end
    Y=train1(:,1);
    
    X_test=ones(size(test1,1),1);
    for col=2:size(test1,2)
        X_test=horzcat(X_test,test1(:,col));
    end
    Y_test=test1(:,1);
    KK=(Kmatrix(X,a,b,sigma,which)+lambda*eye(size(X,1),size(X,1)));
    %assignin('base', 'KK', KK);
    %assignin('base', 'inv_KK', inv(KK));
    modelPrediction=transpose(Y)*(KK\kernel_all(X,X_test,a,b,sigma,which));
    modelPrediction=modelPrediction';
    
    %[Y_test,modelPrediction]
    error=0;
    %horzcat(Y_test, modelPrediction)
    error=sum(power((Y_test-modelPrediction),2));
    error=error/size(Y_test,1);
    
end

function [K]= Kmatrix(X,a,b,sigma,which)
    for row1=1:size(X,1)
        for row2=1:size(X,1)
            K(row1,row2)=kernel(X(row1,:),X(row2,:),a,b,sigma,which);
        end
    end
end

% takes two rows as input!
function [x] = kernel(x1,x2,a,b,sigma,which)
    if (which==1)
        x=x1*transpose(x2);
    elseif (which ==2)
        x=power((x1*transpose(x2)+a),b);
    else
        x=exp(-norm(x1-x2)/(sigma));
    end
          
end

function [K]=kernel_all(X,X_test,a,b,sigma,which)
    for row=1:size(X,1)
        for rcol=1:size(X_test,1)
            K(row,rcol)=kernel(X(row,:),X_test(rcol,:),a,b,sigma,which);
        end
    end
end

function [avgerror]=linearRigidRegression(train,plotNo)
    breakpoint=0.8;
    totalSplit=5;
    lambda=[0,10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3];
    
    errorLambda=zeros(size(lambda,2),1);
    for l=1:size(lambda,2)
        testingError=0;
        for eachSplit=0:4
            [train1,test1]=getTrainandTest(train,eachSplit,totalSplit);
            [error]=getSplitError(train1,test1,lambda(l));
            testingError=testingError+error;
        end
        errorLambda(l)=testingError/totalSplit;
    end
    subplot(5,2,plotNo);
    lambda=(lambda');
    table(lambda,errorLambda)
    lambda=log10(lambda);
    plot(lambda,errorLambda);
    title('Optimal lambda selection');
    xlabel('log10(lambda)');
    ylabel('error');
    avgerror=(mean(errorLambda));
end

function [error]=getSplitError(train1,test1,lambda)
    X=ones(size(train1,1),1);
    for col=2:size(train1,2)
        X=horzcat(X,train1(:,col));
    end
    Y=train1(:,1);
    theta=(transpose(X)*X + lambda*eye(size(X,2),size(X,2)))\(transpose(X)*Y);
    X_test=ones(size(test1,1),1);
    
    for col=2:size(test1,2)
        X_test=horzcat(X_test,test1(:,col));
    end
    Y_test=test1(:,1);
    modelPrediction=X_test*theta;
    error=0;
    
    %horzcat(Y_test, modelPrediction)
    for row=1:size(Y_test,1)
        error=error+power((Y_test(row)-modelPrediction(row)),2);
    end
    
    error=error/size(Y_test,1);
end

function [train1,test1]=getTrainandTest(train,splitNo,totalSlit)
    rTest=1;
    rTrain=1;
    for row=1:size(train,1)
        indexStart=int64(size(train,1)*(splitNo/totalSlit))+1;
        indexEnd=int64(size(train,1)*((splitNo+1)/totalSlit));
        if(row<=indexEnd && row>=indexStart)
            test1(rTest,:)=train(row,:);
            rTest=rTest+1;
        else
            train1(rTrain,:)=train(row,:);
            rTrain=rTrain+1;
        end   
    end
    %size(test1)
    %size(train1)
end

function [train,test]= dataSplit(percent,data)
    lenTrain=int64(percent*(size(data,1)));
    keySet=zeros(size(data,1),1);
    valueSet=zeros(size(data,1),1);
    mapObj = containers.Map(keySet,valueSet);
    
    
    q=1;
    
    while(q<=lenTrain)
        num=randi(size(data,1),1);
        if isKey(mapObj,num)~=1
            mapObj(num) = 1;
            train(q,:)=data(num,:);
            q=q+1;
        end
    end
    q=1;
    for i=1:size(data,1)
        if isKey(mapObj,i)~=1
            test(q,:)=data(i,:);
            q=q+1;
        end
    end

    % normalize | dont normalize first col.
    
    for col=2:size(train,2)
        m(col)=mean(train(:,col));
        s(col)=std(train(:,col));
        for row=1:size(train,1)
            train(row,col)=(train(row,col)-m(col))/s(col);
        end
        
        for row=1:size(test,1)
            test(row,col)=(test(row,col)-m(col))/s(col);
        end
        
    end
    
end