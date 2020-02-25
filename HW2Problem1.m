% HW2 Problem 1
% Created on: Feb 17, 2020
% Author: Chenglong Lin
% Student ID: 001024390
clear all, close all,

mu(:,1) = [-2;0]; mu(:,2) = [2;0];
Sigma(:,:,1) = [1 -0.9;-0.9 2]; Sigma(:,:,2) = [2 0.9;0.9 1];
p = [0.9 0.1]; % class priors for labels 0 and 1 respectively

n = 2; % number of feature dimensions
N1 = 10; % number of iid samples for D10t
N2 = 100; % number of iid samples for D100t
N3 = 1000; % number of iid samples for D1000t
N4 = 10000; % number of iid samples for D10000v

label1 = rand(1,N1) >= p(1); % generate N1 number of sample and determine their labels
label2 = rand(1,N2) >= p(1);
label3 = rand(1,N3) >= p(1);
label4 = rand(1,N4) >= p(1);

Nc1 = [length(find(label1==0)),length(find(label1==1))]; % number of samples from each class, Nc(1) is # of label 0, Nc(2) is # of label 1.
Nc2 = [length(find(label2==0)),length(find(label2==1))];
Nc3 = [length(find(label3==0)),length(find(label3==1))];
Nc4 = [length(find(label4==0)),length(find(label4==1))];

D10t = zeros(n,N1); % save up space, 2x10 of zeros (nxN)
D100t = zeros(n,N2);
D1000t = zeros(n,N3);
D10000v = zeros(n,N4);

lambda = [0 1;1 0]; % loss value
% Draw samples from each class pdf
for L = 0:1
    D10t(:,label1==L) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc1(L+1))'; % store generated column vector in where label is 0 or 1, very compact. Each column in x is a mvrnd vector.
    D100t(:,label2==L) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc2(L+1))';
    D1000t(:,label3==L) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc3(L+1))';
    D10000v(:,label4==L) = mvnrnd(mu(:,L+1),Sigma(:,:,L+1),Nc4(L+1))';
end

%% Problem 1 Part 1
disp("Problem 1 Part 1:");

discriminantScore = log(evalGaussian(D10000v,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(D10000v,mu(:,1),Sigma(:,:,1)));
sortedScores = sort(discriminantScore);
tau = zeros(1+length(sortedScores));
epsilon = 0.01;
tau(1) = sortedScores(1) - epsilon; % min of tau
tau(length(tau)) = sortedScores(length(sortedScores)) + epsilon; % max of tau
for i = 2:length(tau)-1
    tau(i) = (sortedScores(i) + sortedScores(i-1))/2;
end

ROC = zeros(3,length(tau));
for i = 1:length(tau)
    decision = (discriminantScore >= tau(i));
    ind00 = find(decision==0 & label4==0); % index of 00, correct decision on 0 label
    p00 = length(ind00)/Nc4(1); % probability of true negative, P(x=0|L=0)
    ind10 = find(decision==1 & label4==0);
    p10 = length(ind10)/Nc4(1); % probability of false positive
    ind01 = find(decision==0 & label4==1);
    p01 = length(ind01)/Nc4(2); % probability of false negative
    ind11 = find(decision==1 & label4==1);
    p11 = length(ind11)/Nc4(2); % probability of true positive
    ROC(:,i) = [p11; p10; tau(i)];
end

gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); % threshold
tauMin = log(gamma);
decision = (discriminantScore >= tauMin);
ind00 = find(decision==0 & label4==0); % index of 00, correct decision on 0 label
p00 = length(ind00)/Nc4(1); % probability of true negative, P(x=0|L=0)
ind10 = find(decision==1 & label4==0);
p10 = length(ind10)/Nc4(1); % probability of false positive
ind01 = find(decision==0 & label4==1);
p01 = length(ind01)/Nc4(2); % probability of false negative
ind11 = find(decision==1 & label4==1);
p11 = length(ind11)/Nc4(2); % probability of true positive
pError = [p10,p01]*Nc4'/N4; % probability of error, empirically estimated
fprintf('Theoretical estimate of the min-P(error) using expected risk minimization: %.2f%%\n',pError*100);

figure(1),
plot(ROC(2,:),ROC(1,:)); hold on,
plot(p10,p11,'og'); hold on,
title('Detection vs. false alarm ROC - ERM'),
xlabel('False alarm'), ylabel('Detection'),
legend('Detection vs. false alarm','Minimun probability of error point'),

figure(2),
plot3(ROC(2,:),ROC(1,:),ROC(3,:)); hold on,
plot3(p10,p11,tauMin,'og'); hold on,
title('Tau vs. detection vs. false alarm ROC - ERM'),
xlabel('False alarm'), ylabel('Detection'), zlabel('Tau')
legend('Tau vs. detection vs. false alarm','Minimun probability of error point'),

figure(3), % class 0 circle, class 1 +, correct green, incorrect red
plot(D10000v(1,ind00),D10000v(2,ind00),'og'); hold on,
plot(D10000v(1,ind10),D10000v(2,ind10),'or'); hold on,
plot(D10000v(1,ind01),D10000v(2,ind01),'+r'); hold on,
plot(D10000v(1,ind11),D10000v(2,ind11),'+g'); hold on,
axis equal,

% Draw the decision boundary
horizontalGrid = linspace(floor(min(D10000v(1,:))),ceil(max(D10000v(1,:))),101);
verticalGrid = linspace(floor(min(D10000v(2,:))),ceil(max(D10000v(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
figure(3), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

%% Problem 1 Part 2, logistic linear function
% Logistic Regression
% Initialize fitting parameters
disp("Problem 1 Part 2:");

D10t = [ones(N1, 1) D10t']; %N1 x 3, x
D100t = [ones(N2, 1) D100t']; %N2 x 3, x
D1000t = [ones(N3, 1) D1000t']; %N3 x 3, x

initial_theta1 = zeros(n+1, 1); %3 x 1, weights
initial_theta2 = zeros(n+1, 1); %3 x 1, weights
initial_theta3 = zeros(n+1, 1); %3 x 1, weights

label1=double(label1); % y
label2=double(label2); % y
label3=double(label3); % y

alpha = 0.5; % learning rate

% Compute gradient descent to get theta values
[theta1, cost1] = gradient_descent(D10t, N1, label1', initial_theta1, alpha, 0.06);
[theta2, cost2] = gradient_descent(D100t, N2, label2', initial_theta2, alpha, 0.06);
[theta3, cost3] = gradient_descent(D1000t, N3, label3', initial_theta3, alpha, 0.06);

% Test Classifier
% test d10 training set
plot_linear(label1, Nc1, p, 4, (D10t(:,2:3))', theta1, 'Classifier from D10 training set on D10 training samples');
title('Training Data linear regression Classification - D10 training set');
error1 = plot_linear(label4, Nc4, p, 5, D10000v, theta1, 'Classifier from D10 training set on D10000 validation samples');
title('Test Data linear regression Classification - D10 training set');
fprintf('D10 training linear classifier error: %.2f%%\n',error1);

% test d100 training set
plot_linear(label2, Nc2, p, 6, (D100t(:,2:3))', theta2, 'Classifier from D100 training set on D100 training samples');
title('Training Data linear regression Classification - D100 training set');
error2 = plot_linear(label4, Nc4, p, 7, D10000v, theta2, 'Classifier from D100 training set on D10000 validation samples');
title('Test Data linear regression Classification - D100 training set');
fprintf('D100 training linear classifier error: %.2f%%\n',error2);

% test d1000 training set
plot_linear(label3, Nc3, p, 8, (D1000t(:,2:3))', theta3, 'Classifier from D1000 training set on D1000 training samples');
title('Training Data linear regression Classification - D1000 training set');
error3 = plot_linear(label4, Nc4, p, 9, D10000v, theta3, 'Classifier from D1000 training set on D10000 validation samples');
title('Test Data linear regression Classification - D1000 training set');
fprintf('D1000 training linear classifier error: %.2f%%\n',error3);

%Plot cost function
% figure(10); plot(cost1);
% title('Calculated Cost - d10 training set');
% xlabel('Iteration number'); ylabel('Cost');
% 
% figure(11); plot(cost2);
% title('Calculated Cost - d100 training set');
% xlabel('Iteration number'); ylabel('Cost');
% 
% figure(12); plot(cost3);
% title('Calculated Cost - d1000 training set');
% xlabel('Iteration number'); ylabel('Cost');

%% Problem 1 Part 3, logistic quadratic function
disp("Problem 1 Part 3:");

D10t = [D10t D10t(:,2).*D10t(:,2) D10t(:,2).*D10t(:,3) D10t(:,3).*D10t(:,3)]; %N3 x 6, x
D100t = [D100t D100t(:,2).*D100t(:,2) D100t(:,2).*D100t(:,3) D100t(:,3).*D100t(:,3)]; %N3 x 6, x
D1000t = [D1000t D1000t(:,2).*D1000t(:,2) D1000t(:,2).*D1000t(:,3) D1000t(:,3).*D1000t(:,3)]; %N3 x 6, x

initial_theta1 = [initial_theta1; zeros(n+1, 1)]; %6 x 1, weights
initial_theta2 = [initial_theta2; zeros(n+1, 1)]; %6 x 1, weights
initial_theta3 = [initial_theta3; zeros(n+1, 1)]; %6 x 1, weights

[theta1, cost1] = gradient_descent(D10t, N1, label1', initial_theta1, alpha, 0.05);
[theta2, cost2] = gradient_descent(D100t, N2, label2', initial_theta2, alpha, 0.05);
[theta3, cost3] = gradient_descent(D1000t, N3, label3', initial_theta3, alpha, 0.05);

% test d10 training set
plot_quadratic(label1, Nc1, p, 13, (D10t(:,2:3))', theta1, 'Classifier from D10 training set on D10 training samples');
title('Training Data quadratic regression Classification - D10 training set');
error1 = plot_quadratic(label4, Nc4, p, 14, D10000v, theta1, 'Classifier from D10 training set on D10000 validation samples');
title('Test Data quadratic regression Classification - D10 training set');
fprintf('D10 training quadratic classifier error: %.2f%%\n',error1);

% test d100 training set
plot_quadratic(label2, Nc2, p, 15, (D100t(:,2:3))', theta2, 'Classifier from D100 training set on D100 training samples');
title('Training Data quadratic regression Classification - D100 training set');
error2 = plot_quadratic(label4, Nc4, p, 16, D10000v, theta2, 'Classifier from D100 training set on D10000 validation samples');
title('Test Data quadratic regression Classification - D100 training set');
fprintf('D100 training quadratic classifier error: %.2f%%\n',error2);

% test d1000 training set
plot_quadratic(label3, Nc3, p, 17, (D1000t(:,2:3))', theta3, 'Classifier from D1000 training set on D1000 training samples');
title('Training Data quadratic regression Classification - D1000 training set');
error3 = plot_quadratic(label4, Nc4, p, 18, D10000v, theta3, 'Classifier from D1000 training set on D10000 validation samples');
title('Test Data quadratic regression Classification - D1000 training set');
fprintf('D1000 training quadratic classifier error: %.2f%%\n',error3);

%Plot cost function
figure(19); plot(cost1);
title('Calculated Cost quadratic - d10 training set');
xlabel('Iteration number'); ylabel('Cost');

figure(20); plot(cost2);
title('Calculated Cost quadratic- d100 training set');
xlabel('Iteration number'); ylabel('Cost');

figure(21); plot(cost3);
title('Calculated Cost quadratic - d1000 training set');
xlabel('Iteration number'); ylabel('Cost');

%% Functions
function cost = cost_func(theta, x, label,N) 
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h)))); % negative-log-likelihood loss function
end

function error = plot_quadratic(label, Nc, p, fig, x, theta, dataset)
    decision = theta(6).*(x(2,:).^2) + theta(5).*x(1,:).*x(2,:) + theta(4).*(x(1,:).^2) + theta(3).*x(2,:) + theta(2).*x(1,:) + theta(1) > 0; %compare it with 0, if < 0, classify as 0, if > 0, classify as 1
     
    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive
    error = (p10*p(1) + p01*p(2))*100;
    
    % Plot decisions and decision boundary
    figure(fig);
    plot(x(1,ind00),x(2,ind00),'og'); hold on,
    plot(x(1,ind10),x(2,ind10),'or'); hold on,
    plot(x(1,ind01),x(2,ind01),'+r'); hold on,
    plot(x(1,ind11),x(2,ind11),'+g'); hold on,
    f = @(x,y) theta(6).*(y.^2) + theta(5).*x.*y + theta(4).*(x.^2) + theta(3).*y + theta(2).*x + theta(1);
    fimplicit(f, [min(x(1,:))-10, max(x(1,:))+10, min(x(2,:))-10, max(x(2,:))+10]);
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions',dataset);
end

function error = plot_linear(label, Nc, p, fig, x, theta, dataset)
    % Choose points to draw boundary line
    plot_x1 = [min(x(1,:))-2,  max(x(2,:))+2]; % use boundary x1 to draw decision line             
    plot_x2 = (-1./theta(3)).*(theta(2).*plot_x1 + theta(1)); % y = [w2 w3][x1 x2]' + w1, because the y axis is not drawn, set y=0, then x2 = (-w2x1-w1)/w3

    % Coefficients for decision boundary line equation
    coeff = polyfit([plot_x1(1), plot_x1(2)], [plot_x2(1,1), plot_x2(1,2)], 1); % find coeff of the decision line, [coeff_x^n coeff_x^n-1 coeff_x^n-2 ...]

    % Decide based on which side of the line each point is on
    if coeff(1) >= 0 % if coeff(1) is postive, then line drawn is from the bottom to the top
        decision = (coeff(1).*x(1,:) + coeff(2)) > x(2,:); % if x2 on the right of the line with respect to line drawn from bottom to the top, d=1;
    else % if coeff(1) is negative, then line drawn is from the top to the bottom
        decision = (coeff(1).*x(1,:) + coeff(2)) < x(2,:); % the decision is reversed
    end

    ind00 = find(decision==0 & label==0); % true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % false negative
    ind11 = find(decision==1 & label==1); % true positive
    error = (p10*p(1) + p01*p(2))*100;

    % Plot decisions and decision boundary
    figure(fig);
    plot(x(1,ind00),x(2,ind00),'og'); hold on,
    plot(x(1,ind10),x(2,ind10),'or'); hold on,
    plot(x(1,ind01),x(2,ind01),'+r'); hold on,
    plot(x(1,ind11),x(2,ind11),'+g'); hold on,
    plot(plot_x1, plot_x2); % draw (x1_min,x2_min) to (x1_max, x2_max)
    axis([plot_x1(1), plot_x1(2), min(x(2,:))-2, max(x(2,:))+2]) % limits for the current axes min(x1) to max(x1), min(x2) to max(x2)
    legend('Class 0 Correct Decisions','Class 0 Wrong Decisions','Class 1 Wrong Decisions','Class 1 Correct Decisions',dataset);
end

function [theta, cost] = gradient_descent(x, N, label, theta, alpha, epsilon) %alpha = learning rate; epsilon = threshold
    h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function
    cost = [1 (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))))];
    cost_gradient = (1/N)*(x' * (h - label));
    theta = theta - (alpha.*cost_gradient); % Update theta
    while abs(cost(end-1) - cost(end))*100 / cost(end-1) > epsilon % while cost change is small
        h = 1 ./ (1 + exp(-x*theta));	% Sigmoid function   
        cost = [cost (-1/N)*((sum(label' * log(h)))+(sum((1-label)' * log(1-h))))]; % when label is 0 use 1-h, use h when label is 1
        cost_gradient = (1/N)*(x' * (h - label));
        theta = theta - (alpha.*cost_gradient); % Update theta
    end
end