function a5_20027306
% Function for CISC271, Winter 2021, Assignment #5

% Read the test data from a CSV file, standardize, and extract labels;
% for the "college" data and Fisher's Iris data

Xraw = csvread('collegenum.csv',1,1);
[~, Xcoll] = pca(zscore(Xraw(:,2:end)), 'NumComponents', 2);
ycoll = round(Xraw(:,1)>0);

load fisheriris;
Xiris = zscore(meas);
yiris = ismember(species,'setosa');

% Call the functions for the questions in the assignment
a5q1(Xcoll, ycoll);
a5q2(Xiris, yiris);

% END FUNCTION
end

function a5q1(Xmat, yvec)
% A5Q1(XMAT,YVEC) solves Question 1 for data in XMAT with
% binary labels YVEC
%
% INPUTS:
%         XMAT - MxN array, M observations of N variables
%         YVEC - Mx1 binary labels, interpreted as >=0 or <0
% OUTPUTS:
%         none

% Augment the X matrix with a 1's vector
Xaug = [Xmat ones(size(Xmat, 1), 1)];

% Perceptron initialization and estimate of hyperplane
eta = 0.001;
[v_ann ix] = sepbinary(Xaug, yvec, eta);
v_ann = v_ann/norm(v_ann);

% Logistic regression estimate of hyperplane
v_log = logreg(Xmat, yvec);
v_log = v_log/norm(v_log);

% Score the data using the hyperplane augmented vectors
z_ann = Xaug*v_ann;
z_log = Xaug*v_log;

% Find the ROC curves
[px_ann, py_ann, ~, auc_ann] = perfcurve(yvec, z_ann, +1);
[px_log, py_log, ~, auc_log] = perfcurve(yvec, z_log, +1);

% %
% % STUDENT CODE GOES HERE: compute the accuracies of hyperplanes
% % and display the results to the command console
% %
%Area under the curve and accuracy values
auc_ann
[acc_ann, bestThreshold_ann] = accuracy(yvec, z_ann)
auc_log
[acc_log, bestThreshold_log] = accuracy(yvec, z_log)

% %
% % STUDENT CODE GOES HERE: plot figures and display results
% %

% ROC curves
figure(1);
plotRoc(px_ann, py_ann, 'Perceptron ROC', 'FPR','TPR')

figure(2);
plotRoc(px_log, py_log, 'Logistic regression ROC', 'FPR','TPR')

% Scatter plots and separating lines
%Plots of dimensionally reduced data plus the separating hyperplane for
%each method.
figure(3);
plotClusters(Xmat(:,1), Xmat(:,2), yvec, 'Perceptron hyperplane', 'X1', 'X2','rb')
%Add separating hyperplane
hold on
plotline(v_ann, 'k');
hold off

figure(4);
plotClusters(Xmat(:,1), Xmat(:,2), yvec, 'Logistic regression hyperplane', 'X1', 'X2','mc')
%Add separating hyperplane
hold on
plotline(v_log, 'k');
hold off

% END FUNCTION
end

function [acc, bestThreshold] = accuracy(yvec_in,zvec_in)
% Sort the scores and permute the labels accordingly
[zvec zndx] = sort(zvec_in);
yvec = yvec_in(zndx);

% Sort and find a unique subset of the scores; problem size
bvec = unique(zvec);
bm = numel(bvec);

bestThreshold = -inf;
acc = -inf;
for ix = 1:bm
    cmat = confmat(yvec,zvec,bvec(ix,:));
    TP = cmat(1,1);
    FP = cmat(2,1);
    FN = cmat(1,2);
    TN = cmat(2,2);
    P = TP + FN;
    N = FP + TN;
    cacc = (TP + TN) / (P + N);
    if cacc > acc
        acc = cacc;
        bestThreshold = bvec(ix,:);
    end
end
end

function cmat = confmat(yvec, zvec, theta)
% CMAT=CONFMAT(YVEC,ZVEC,THETA) finds the confusion matrix CMAT for labels
% YVEC from scores ZVEC and a threshold THETA. YVEC is assumed to be +1/-1
% and each entry of ZVEC is scored as -1 if <THETA and +1 otherwise. CMAT
% is returned as [TP FN ; FP TN]
%
% INPUTS:
%         YVEC  - Mx1 values, +/- computed as ==/~= 1
%         ZVEC  - Mx1 scores, real numbers
%         THETA - threshold real-valued scalar
% OUTPUTS:
%         CMAT  - 2x2 confusion matrix; rows are +/- labels,
%                 columns are +/- classifications

% Find either 1 or 0 vector of quantizations
qvec = zvec >= theta;

% Compute the confusion matrix by entries
% %
% % STUDENT CODE GOES HERE: COMPUTE MATRIX
% %
TP = 0;
TN = 0;
FP = 0;
FN = 0;

[numberOfObservations,~] = size(yvec);
for ix = 1:numberOfObservations
    if (qvec(ix,:) == yvec(ix,:))
        if(qvec(ix,:) == 1)
            TP = TP + 1;
        else
            TN = TN + 1;
        end
    elseif (qvec(ix,:) ~= yvec(ix,:))
        if(qvec(ix,:) == 1)
            FP = FP + 1;
        else
            FN = FN + 1;
        end
    end
end
cmat = [TP FN ; FP TN];
end

function plotRoc(xVal,yVal,figureName,xlabelName,ylabelName)
plot(xVal,yVal,'o')
hold on
plot(0:0.01:1,0:0.01:1,'--k')
hold off
xlabel(xlabelName)
ylabel(ylabelName)
title(figureName)
end

function plotClusters(xVal,yVal,group,figureName,xlabelName,ylabelName,color)
gscatter(xVal,yVal,group,color,'.')
xlabel(xlabelName)
ylabel(ylabelName)
title(figureName)
end

function [v_final, i_used] = sepbinary(Xmat, yvec, eta_in)
% [V_FINAL,I_USED]=LINSEPLEARN(VINIT,ETA,XMAT,YVEC)
% uses the Percetron Algorithm to linearly separate training vectors
% INPUTS:
%         ZMAT    - Mx(N+1) augmented data matrix
%         YVEC    - Mx1 desired classes, 0 or 1
%         ETA     - optional, scalar learning rate, default is 1
% OUTPUTS:
%         V_FINAL - (N+1)-D new estimated weight vector
%         I_USED  - scalar number of iterations used
% ALGORITHM:
%         Vectorized form of perceptron gradient descent

% Use optional argument if it is preset
if nargin>=3 && exist('eta_in') && ~isempty(eta_in)
    eta = eta_in;
else
    eta = 1;
end

% Internal constant: maximum iterations to use
imax = 10000;

% Initialize the augmented weight vector as a 1's vector
v_est = ones(size(Xmat, 2), 1);

% Loop a limited number of times
for i_used=0:imax
    missed = 0;
    % Assume that the final weight is the current estimate
    v_final = v_est;
    rvec = zeros(size(yvec));
    
    % %
    % % STUDENT CODE GOES HERE: compute the Perceptron update
    % %
    %Augmented data matrix times the augmented weight vector
    zbyW = Xmat*v_est;
    %Heaviside step function. Gives quantizations.
    qvec = heaviside(zbyW);
    %Residual error between label and quantization
    rvec = yvec - qvec;
    % Update using the learning rate eta with a batch update
    v_est = v_est + eta*(Xmat'*rvec);
    
    % Continue looping if any data are mis-classified
    missed = norm(rvec, 1) > 0;
    
    % Stop if the current estimate has converged
    if (missed==0)
        v_final = v_est;
        break;
    end
end

% END FUNCTION
end

function a5q2(Xmat, yvec)
% A5Q2(XMAT,YVEC) solves Question 2 for data in XMAT with
% binary labels YVEC
%
% INPUTS:
%         XMAT - MxN array, M observations of N variables
%         YVEC - Mx1 binary labels, interpreted as ~=0 or ==0
% OUTPUTS:
%         none

% Anonymous function: centering matrix of parameterized size
Gmat =@(k) eye(k) - 1/k*ones(k,k);

% Problem size
[m, n] = size(Xmat);

% Default projection of data
Mgram = Xmat(:, 1:2);

% Reduce data to Lmax-D; here, to 2D
Lmax = 2;

% Set an appropriate gamma for a Gaussian kernel
sigma2 = 2*m;

% Compute the centered MxM Gram matrix for the data
Kmat = Gmat(m)*gramgauss(Xmat, sigma2)*Gmat(m);

% %
% % STUDENT CODE GOES HERE: Compute Kernel PCA
% %
% Kernel PCA uses the spectral decomposition of the Gram matrix;
% sort the eigenvalues and eigenvectors in descending order and
% project the Gram matrix to "Lmax" dimensions

%Collect the eigenvectors and eigenvalues from the centered Gram matrix.
[eigenvectors, eigenvalues] = eig(Kmat, 'vector');
%Sort the eigenvalues and corresponding eigenvectors in descending order
[~, lx] = sort(eigenvalues, 'descend');
%The eigenvectors in descending order. May not look like they are in
%decending order.
eigenvectors = eigenvectors(:,lx);
%Get the first Lmax eigenvectors
qvec = eigenvectors(:,1:Lmax);
%Rewrite projection of data with kernel PCA score vector
Mgram = Kmat*qvec;

% Cluster the first two dimensions of the projection as 0,+1
rng('default');
yk2 = kmeans(Mgram, 2) - 1;

% %
% % STUDENT CODE GOES HERE: plot and display results to console
% %

% Plot the labels and the clusters
%Plot with projected data and classificaitons from labels and k-means 
figure(5);
plotClusters(Mgram(:,1), Mgram(:,2), yvec, 'Iris data with labels', 'Projected data column 1', 'Projected data column 2','rb')

figure(6);
plotClusters(Mgram(:,1), Mgram(:,2), yk2, 'Iris data with K-means clusters', 'Projected data column 1', 'Projected data column 2','mc')

% END FUNCTION
end

function Kmat = gramgauss(Xmat, sigma2_in)
% K=GRAMGAUSS(X,SIGMA2)computes a Gram matrix for data in X
% using the Gaussian exponential exp(-1/sigma2*norm(X_i - X_j)^2)
%
% INPUTS:
%         X      - MxN data with M observations of N variables
%         sigma2 - optional scalar, default value is 1
% OUTPUTS:
%         K       NxN Gram matrix

% Optionally use the provided sigma^2 scalar
if (nargin>=2) & ~isempty('sigma2_in')
    sigma2 = sigma2_in;
else
    sigma2 = 1;
end

% Default Gram matrix is the standardized Euclidean distance
Kmat = pdist2(Xmat, Xmat, 'seuclidean');

% %
% % STUDENT CODE GOES HERE: compute the Gram matrix
% %

%Gaussian Kernel function for row spaces. Uses sigma2 which is the variance
%to be used.
gaussianKernel = @(X_i,X_j) exp(-1/sigma2*norm(X_i - X_j)^2);
%Size of original matrix to construct Gram matrix
[numObservations, ~] = size(Xmat);

%Create Gram matrix, end result is a 150 x 150 Gram matrix
for ix = 1:numObservations
    for jx = 1:numObservations
        %Apply the kernel function on location ij composed of two rows
        Kmat(ix,jx) = gaussianKernel(Xmat(ix,:),Xmat(jx,:));
    end
end
end

% %
% % NO STUDENT CHANGES NEEDED BELOW HERE
% %
function waug = logreg(Xmat,yvec)
% WAUG=LOGREG(XMAT,YVEC) performs binary logistic regression on data
% matrix XMAT that has binary labels YVEC, using GLMFIT. The linear
% coefficients of the fit are in vector WAUG. Important note: the
% data XMAT are assumed to have no intercept term because these may be
% standardized data, but the logistic regression coefficients in WAUG
% will have an intercept term. The labels in YVEC are managed by
% >0 and ~>0, so either (-1,+1) convention or (0,1) convention in YVEC
% are acceptable.
%
% INPUTS:
%         XMAT - MxN array, of M observations in N variables
%         YVEC - Mx1 vector, binary labels
% OUTPUTS:
%         WAUG - (N+1)x1 vector, coefficients of logistic regression

% Perform a circular shift of the GLMFIT coefficients so that
% the final coefficient acts as an intercept term for XMAT

warnstate = warning('query', 'last');
warning('off');
waug = circshift(glmfit(Xmat ,yvec>0, ...
    'binomial', 'link', 'probit'), -1);
warning(warnstate);

% END FUNCTION
end

function ph = plotline(vvec, color, lw, nv)
% PLOTLINE(VVEC,COLOR,LW,NV) plots a separating line
% into an existing figure
% INPUTS:
%        VVEC   - (M+1) augmented weight vector
%        COLOR  - character, color to use in the plot
%        LW   - optional scalar, line width for plotting symbols
%        NV   - optional logical, plot the normal vector
% OUTPUT:
%        PH   - plot handle for the current figure
% SIDE EFFECTS:
%        Plot into the current window.

% Set the line width
if nargin >= 3 & ~isempty(lw)
    lwid = lw;
else
    lwid = 2;
end

% Set the normal vector
if nargin >= 4 & ~isempty(nv)
    do_normal = true;
else
    do_normal = false;
end

% Current axis settings
axin = axis();

% Scale factor for the normal vector
sval = 0.025*(axin(4) - axin(3));

% Four corners of the current axis
ll = [axin(1) ; axin(3)];
lr = [axin(2) ; axin(3)];
ul = [axin(1) ; axin(4)];
ur = [axin(2) ; axin(4)];

% Normal vector, direction vector, hyperplane scalar
nlen = norm(vvec(1:2));
uvec = vvec/nlen;
nvec = uvec(1:2);
dvec = [-uvec(2) ; uvec(1)];
bval = uvec(3);

% A point on the hyperplane
pvec = -bval*nvec;

% Projections of the axis corners on the separating line
clist = dvec'*([ll lr ul ur] - pvec);
cmin = min(clist);
cmax = max(clist);

% Start and end are outside the current plot axis, no problem
pmin = pvec +cmin*dvec;
pmax = pvec +cmax*dvec;

% Create X and Y coordinates of a box for the current axis
xbox = [axin(1) axin(2) axin(2) axin(1) axin(1)];
ybox = [axin(3) axin(3) axin(4) axin(4) axin(3)];

% Intersections of the line and the box
[xi, yi] = polyxpoly([pmin(1) pmax(1)], [pmin(2) pmax(2)], xbox, ybox);

% Point midway between the intersections
pmid = [mean(xi) ; mean(yi)];

% Range of the intersection line
ilen = 0.5*norm([(max(xi) - min(xi)) ; (max(yi) - min(yi))]);

% Plot the line according to the color specification
hold on;
if ischar(color)
    ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
        [color '-'], 'LineWidth', lwid);
else
    ph = plot([pmin(1) pmax(1)], [pmin(2) pmax(2)], ...
        'Color', color, 'LineStyle', '-', 'LineWidth', lwid);
end
if do_normal
    quiver(pmid(1), pmid(2), nvec(1)*ilen*sval, nvec(2)*ilen*sval, ...
        'Color', color, 'LineWidth', lwid, ...
        'MaxHeadSize', ilen/2, 'AutoScale', 'off');
end
hold off;

% Remove this label from the legend, if any
ch = get(gcf,'children');
for ix=1:length(ch)
    if strcmp(ch(ix).Type, 'legend')
        ch(ix).String{end} = '';
    end
end

% END FUNCTION
end