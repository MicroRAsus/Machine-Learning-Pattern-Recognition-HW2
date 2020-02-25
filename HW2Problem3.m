% HW2 Problem 3
% Created on: Feb 17, 2020
% Author: Chenglong Lin
% Student ID: 001024390
%% D1000
clear all, close all,
delta = 0.3; % tolerance for EM stopping criterion
regWeight = 1e-5; % regularization parameter for covariance estimates
N = [10 100 1000]; % number of sample in each data set
NumExperiment = 100;
NumSampling = 720; % max component we are trying to estimate is 6, so we need to estimate 36 parameters, we use 20 samples to estimate each parameter, so 720 samples are needed.
sigmaV = 1e-4.*eye(2);
MaxIte = 100;
B = 10;

% 4-component GMM true knowledge
alpha_true = [0.24,0.23,0.26,0.27]; % alpha is prior
mu_true = [-5 2 6 0;0 -5 3 8];
Sigma_true(:,:,1) = [3 1;1 5];
Sigma_true(:,:,2) = [5 1;1 4];
Sigma_true(:,:,3) = [4 1;1 5];
Sigma_true(:,:,4) = [3 1;1 3];

[d,~] = size(mu_true); % get dimension
candidate = [1 2 3 4 5 6]; % candidate GMM component number

winnerCountD10 = zeros(1, length(candidate));
winnerCountD100 = zeros(1, length(candidate));
winnerCountD1000 = zeros(1, length(candidate));

%% D1000
redo = zeros(NumExperiment, B, size(candidate,2)); %reinitialize counter
for a = 1:NumExperiment
    D1000 = randGMM(N(3),alpha_true,mu_true,Sigma_true);
    
    LLH = zeros(B,size(candidate,2)); %loglikelihood from experiments
    for b=1:B
        Dtrain = D1000(:,randi([1 N(3)],1,NumSampling)) + mvnrnd([0;0],sigmaV,NumSampling)'; %training set with gaussian noise, sample with replacement
        Dval = D1000(:,randi([1 N(3)],1,NumSampling)); %test set
        
        clear temp;
        for M = candidate
            alpha = ones(1,M)/M; % assume gaussian components are uniformly selected
            Converged = 0; % Not converged at the beginning
            while ~Converged %if not converged after max ite, reinitialize
                shuffledIndices = randperm(size(Dtrain,2)); % randperm?, N is number of samples drawn from GMM
                mu = Dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
                [~,assignedCentroidLabels] = min(pdist2(mu',Dtrain'),[],1); % assign each sample to the nearest mean
                for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
                    Sigma(:,:,m) = cov(Dtrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
                end
                
%                 t = 0; %displayProgress(t,x,alpha,mu,Sigma);
                
                ite = 0;
                while ~Converged & ite <= MaxIte % if not converged and not exceeds max ite, run
                    for l = 1:M
                        temp(l,:) = repmat(alpha(l),1,size(Dtrain,2)).*evalGaussian(Dtrain,mu(:,l),Sigma(:,:,l)); %prior * pdf
                    end
                    plgivenx = temp./sum(temp,1); % sum of each column
                    alphaNew = mean(plgivenx,2); % mean of each row, new priors
                    w = plgivenx./repmat(sum(plgivenx,2),1,size(Dtrain,2));
                    muNew = Dtrain*w';
                    for l = 1:M
                        v = Dtrain-repmat(muNew(:,l),1,size(Dtrain,2));
                        u = repmat(w(l,:),d,1).*v;
                        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
                    end
                    Dalpha = sum(abs(alphaNew-alpha'));
                    Dmu = sum(sum(abs(muNew-mu)));
                    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                    ite = ite + 1;
%                     t = t+1;
%                     displayProgress(t,Dtrain,alpha,mu,Sigma);
                end
                if ~Converged
                    redo(a, b, M) = redo(a, b, M) + 1;
                end    
%                 clf(1);
            end
            LLH(b,M) = sum(log(evalGMM(Dval,alpha,mu,Sigma)));
            %fprintf('candidate: %d\n', M);
        end
        %fprintf('b: %d\n', b);
    end
    [~,index] = max(mean(LLH,1));
    winnerCountD1000(index) = winnerCountD1000(index) + 1;
    %fprintf('Component number %d win!\n', index);
end
[~,index] = max(winnerCountD1000);
fprintf('D1000 training EM decision: %d\n', index);
fprintf('D1000 training average reinitialization: %.2f\n', mean(redo,'all'));

%% D100
redo = zeros(NumExperiment, B, size(candidate,2)); %reinitialize counter
for a = 1:NumExperiment
	D100 = randGMM(N(2),alpha_true,mu_true,Sigma_true);
    
    LLH = zeros(B,size(candidate,2)); %loglikelihood from experiments
    for b=1:B
        Dtrain = D100(:,randi([1 N(2)],1,NumSampling)) + mvnrnd([0;0],sigmaV,NumSampling)'; %training set with gaussian noise, sample with replacement
        Dval = D100(:,randi([1 N(2)],1,NumSampling)); %test set
        
        clear temp;
        for M = candidate
            alpha = ones(1,M)/M; % assume gaussian components are uniformly selected
            Converged = 0; % Not converged at the beginning
            while ~Converged %if not converged after max ite, reinitialize
                shuffledIndices = randperm(size(Dtrain,2)); % randperm?, N is number of samples drawn from GMM
                mu = Dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
                [~,assignedCentroidLabels] = min(pdist2(mu',Dtrain'),[],1); % assign each sample to the nearest mean
                for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
                    Sigma(:,:,m) = cov(Dtrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
                end
                
%                 t = 0; %displayProgress(t,x,alpha,mu,Sigma);
                
                ite = 0;
                while ~Converged & ite <= MaxIte % if not converged and not exceeds max ite, run
                    for l = 1:M
                        temp(l,:) = repmat(alpha(l),1,size(Dtrain,2)).*evalGaussian(Dtrain,mu(:,l),Sigma(:,:,l)); %prior * pdf
                    end
                    plgivenx = temp./sum(temp,1); % sum of each column
                    alphaNew = mean(plgivenx,2); % mean of each row, new priors
                    w = plgivenx./repmat(sum(plgivenx,2),1,size(Dtrain,2));
                    muNew = Dtrain*w';
                    for l = 1:M
                        v = Dtrain-repmat(muNew(:,l),1,size(Dtrain,2));
                        u = repmat(w(l,:),d,1).*v;
                        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
                    end
                    Dalpha = sum(abs(alphaNew-alpha'));
                    Dmu = sum(sum(abs(muNew-mu)));
                    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                    ite = ite + 1;
%                     t = t+1;
%                     displayProgress(t,Dtrain,alpha,mu,Sigma);
                end
                if ~Converged
                    redo(a, b, M) = redo(a, b, M) + 1;
                end
%                 clf(1);
            end
            LLH(b,M) = sum(log(evalGMM(Dval,alpha,mu,Sigma)));
            %fprintf('candidate: %d\n', M);
        end
        %fprintf('b: %d\n', b);
    end
    [~,index] = max(mean(LLH,1));
    winnerCountD100(index) = winnerCountD100(index) + 1;
    %fprintf('Component number %d win!\n', index);
end
[~,index] = max(winnerCountD100);
fprintf('D100 training EM decision: %d\n', index);
fprintf('D100 training average reinitialization: %.2f\n', mean(redo,'all'));

%% D10
redo = zeros(NumExperiment, B, size(candidate,2)); %reinitialize counter
for a = 1:NumExperiment
	D10 = randGMM(N(1),alpha_true,mu_true,Sigma_true);
    
    LLH = zeros(B,size(candidate,2)); %loglikelihood from experiments
    for b=1:B
        Dtrain = D10(:,randi([1 N(1)],1,NumSampling)) + mvnrnd([0;0],sigmaV,NumSampling)'; %training set with gaussian noise, sample with replacement
        Dval = D10(:,randi([1 N(1)],1,NumSampling)); %test set
        
        clear temp;
        for M = candidate
            alpha = ones(1,M)/M; % assume gaussian components are uniformly selected
            Converged = 0; % Not converged at the beginning
            while ~Converged %if not converged after max ite, reinitialize
                shuffledIndices = randperm(size(Dtrain,2)); % randperm?, N is number of samples drawn from GMM
                mu = Dtrain(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
                [~,assignedCentroidLabels] = min(pdist2(mu',Dtrain'),[],1); % assign each sample to the nearest mean
                for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
                    Sigma(:,:,m) = cov(Dtrain(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
                end
                
%                 t = 0; %displayProgress(t,x,alpha,mu,Sigma);
                
                ite = 0;
                while ~Converged & ite <= MaxIte % if not converged and not exceeds max ite, run
                    for l = 1:M
                        temp(l,:) = repmat(alpha(l),1,size(Dtrain,2)).*evalGaussian(Dtrain,mu(:,l),Sigma(:,:,l)); %prior * pdf
                    end
                    plgivenx = temp./sum(temp,1); % sum of each column
                    alphaNew = mean(plgivenx,2); % mean of each row, new priors
                    w = plgivenx./repmat(sum(plgivenx,2),1,size(Dtrain,2));
                    muNew = Dtrain*w';
                    for l = 1:M
                        v = Dtrain-repmat(muNew(:,l),1,size(Dtrain,2));
                        u = repmat(w(l,:),d,1).*v;
                        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
                    end
                    Dalpha = sum(abs(alphaNew-alpha'));
                    Dmu = sum(sum(abs(muNew-mu)));
                    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
                    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
                    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
                    ite = ite + 1;
%                     t = t+1;
%                     displayProgress(t,Dtrain,alpha,mu,Sigma);
                end
                if ~Converged
                    redo(a, b, M) = redo(a, b, M) + 1;
                end
%                 clf(1);
            end
            LLH(b,M) = sum(log(evalGMM(Dval,alpha,mu,Sigma)));
            %fprintf('candidate: %d\n', M);
        end
        %fprintf('b: %d\n', b);
    end
    [~,index] = max(mean(LLH,1));
    winnerCountD10(index) = winnerCountD10(index) + 1;
%     fprintf('Component number %d win!\n', index);
end
[~,index] = max(winnerCountD10);
fprintf('D10 training EM decision: %d\n', index);
fprintf('D10 training average reinitialization: %.2f\n', mean(redo,'all'));

%% Functions
function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z = randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

function displayProgress(t,x,alpha,mu,Sigma)
    figure(1),
    if size(x,1)==2
        subplot(1,2,1), cla,
        plot(x(1,:),x(2,:),'b.'); 
        xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
        axis equal, hold on;
        rangex1 = [min(x(1,:)),max(x(1,:))];
        rangex2 = [min(x(2,:)),max(x(2,:))];
        [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
        contour(x1Grid,x2Grid,zGMM); axis equal, 
        subplot(1,2,2), 
    end
    logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
    plot(t,logLikelihood,'b.'); hold on,
    xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
    drawnow; pause(0.1),
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2) % for drawing the GMM components using updatated mu and sigma 
    x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
    x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
    [h,v] = meshgrid(x1Grid,x2Grid);
    GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
    zGMM = reshape(GMM,91,101);
    %figure(1), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
end