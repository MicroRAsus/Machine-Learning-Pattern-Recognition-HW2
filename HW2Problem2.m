% HW2 Problem 2
% Created on: Feb 17, 2020
% Author: Chenglong Lin
% Student ID: 001024390
clear all, close all,

nRealizations = 100; % # of experiments
mu = 0; % mean
sigmaV = 0.5; % noise sigma
B = 5;
gammaArray = logspace(-B,B);
N = 10; % # of samples
w = [1.5 -1.5 -1.5 1.5]'; % 1x4 true parameter [d c b a]T

avMsqError = zeros(nRealizations,length(gammaArray));
for n = 1:nRealizations
    % Draw N samples of x from a Uniform distribution[-1,1]
    x = rand(1,N)*2 - 1; % 1xN
    % Generate noise from normal distribution
    v = normrnd(mu, sigmaV^2, 1, N); %1xN samples of noise for each x sample
    
    % Calculate y
	yTruth{1,n} = yFunc(x,w); % n cell y truth array 1xN
	y = yTruth{1,n} + v; % 1xN
	z = [ones(1,size(x,2)); x; x.^2; x.^3]; % [1 x x^2 x^3]^T, 4xN

    % Compute z*z^T for each sample, third dim is the sample index
	for j = 1:N
        zzT(:,:,j) = z(:,j)*z(:,j)'; % each page is 4x4
	end
    
    % MAP parameter estimation
    for i = 1:length(gammaArray)
        gamma = gammaArray(i);
        thetaMAP{n,i} = (sum(zzT,3)+sigmaV^2/gamma^2*eye(size(z,1)))^-1*sum(repmat(y,size(z,1),1).*z,2);
        avMsqError(n,i) = length(w)\sum((thetaMAP{n,i} - w).^2);
    end
end 

%% Plot results - MAP Ensemble: mean squared error
fig = figure; fig.Position([1,2]) = [50,100];
fig.Position([3 4]) = 1.5*fig.Position([3,4]);
percentileArray = [0,25,50,75,100];

ax = gca; hold on; box on;
prctlMsqError = prctile(avMsqError,percentileArray,1);
p=plot(ax,gammaArray,prctlMsqError,'LineWidth',2);
xlabel('gamma'); ylabel('average mean squared error of parameters'); ax.XScale = 'log';
lgnd = legend(ax,p,[num2str(percentileArray'), repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';
[~,ind] = min(abs(prctlMsqError(3,:)));
plot(ax,gammaArray(ind),prctlMsqError(3,ind),'ro');
lgnd = legend(ax,p,[num2str(percentileArray'), repmat(' percentile',length(percentileArray),1)]); lgnd.Location = 'southwest';

% Draw polynomials
[~,ind] = min(avMsqError(:));
[I,J] = ind2sub([size(avMsqError,1) size(avMsqError,2)],ind);
minTheta = cell2mat(thetaMAP(I,J));
[~,ind] = max(avMsqError(:));
[I,J] = ind2sub([size(avMsqError,1) size(avMsqError,2)],ind);
maxTheta = cell2mat(thetaMAP(I,J));
med = median(avMsqError(:));
[~,ind] = min(abs(avMsqError(:)-med));
[I,J] = ind2sub([size(avMsqError,1) size(avMsqError,2)],ind);
medianTheta = cell2mat(thetaMAP(I,J));
figure(2);
f = @(x) w(4)*x.^3 + w(3)*x.^2 + w(2)*x + w(1);
x = -20:0.1:20;
plot(x,f(x),'g'); hold on,
f = @(x) minTheta(4)*x.^3 + minTheta(3)*x.^2 + minTheta(2)*x + minTheta(1);
plot(x,f(x),'r--'); hold on,
f = @(x) maxTheta(4)*x.^3 + maxTheta(3)*x.^2 + maxTheta(2)*x + maxTheta(1);
plot(x,f(x),'b--'); hold on,
f = @(x) medianTheta(4)*x.^3 + medianTheta(3)*x.^2 + medianTheta(2)*x + medianTheta(1);
plot(x,f(x),'y--'); hold on,
legend('Original polynomial', 'Minimum error MAP estimate', 'Maximum error MAP estimate', 'Median error MAP estimate' ),
xlabel('x'), ylabel('y'),
title('MAP estimated polynomial vs original polynomial');

%% Function to calculate y (without noise), given x and w
function y = yFunc(x,w)
    y = w(4).*x.^3 + w(3).*x.^2 + w(2).*x + w(1); %y = a*x^3 + b*x^2 + c*x + d
end