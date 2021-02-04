% Two class classification example using SVM

% 1. Generate data points (x,y) 100 example for
%----------------------------------------------
% each class
N = 100;
% Class 1
x1 = 4 + randn(N,1);
y1 = 7 + randn(N,1);
class_one = [x1, y1];
% Class 2
x2 = 6 + randn(N,1);
y2 = 4 + randn(N,1);
class_two = [x2, y2];

% 2. Plot the data
%----------------------------------------------
%scatter(x1, y1, 'r')
%xlabel('x');
%ylabel('y');
%hold on
%scatter(x2, y2, 'b')
%hold off

% 3. Fit SVM
%----------------------------------------------
% Concatenate data and create vector of labels
data = [class_one; class_two];
class_one_labels = zeros(N,1) - 1;
class_two_labels = zeros(N,1) + 1;
labels = [class_one_labels; class_two_labels];
% Fit the svm
model = fitcsvm(data, labels);

% 4. Display the results of SVM
%----------------------------------------------
% Find the intercept and gradient of the
% separating boundary
alphas = model.Alpha;
sup_vecs = model.SupportVectors;
betas = model.Beta;
m = -betas(1)/betas(2);
b = -model.Bias/betas(2);

% Plot the line, support vectors and data points
scatter(x1, y1, 'r', 'filled')
xlabel('x');
ylabel('y');

hold on
% Plot class 2 data
scatter(x2, y2, 'b', 'filled')
% Plot separating line
x = linspace(0,10);
y = m.*x + b;
plot(x,y)
% Plot support vectors
plot(sup_vecs(:,1), sup_vecs(:,2), 'ko', 'MarkerSize', 10)
legend('Class 1', 'Class 2', 'Sparating boundary', 'Support Vector')

hold off

%5. Calculate the error rate of the 
%----------------------------------------------
val = crossval(model,'KFold',2);
loss = kfoldLoss(val);
disp('Loss:');
disp(loss);
