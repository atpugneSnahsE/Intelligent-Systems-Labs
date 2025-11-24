% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Features: color + roundness
% Apples
hsv_value_A1=spalva_color(A1);  metric_A1=apvalumas_roundness(A1);
hsv_value_A2=spalva_color(A2);  metric_A2=apvalumas_roundness(A2);
hsv_value_A3=spalva_color(A3);  metric_A3=apvalumas_roundness(A3);
hsv_value_A4=spalva_color(A4);  metric_A4=apvalumas_roundness(A4);
hsv_value_A5=spalva_color(A5);  metric_A5=apvalumas_roundness(A5);
hsv_value_A6=spalva_color(A6);  metric_A6=apvalumas_roundness(A6);
hsv_value_A7=spalva_color(A7);  metric_A7=apvalumas_roundness(A7);
hsv_value_A8=spalva_color(A8);  metric_A8=apvalumas_roundness(A8);
hsv_value_A9=spalva_color(A9);  metric_A9=apvalumas_roundness(A9);

% Pears
hsv_value_P1=spalva_color(P1);  metric_P1=apvalumas_roundness(P1);
hsv_value_P2=spalva_color(P2);  metric_P2=apvalumas_roundness(P2);
hsv_value_P3=spalva_color(P3);  metric_P3=apvalumas_roundness(P3);
hsv_value_P4=spalva_color(P4);  metric_P4=apvalumas_roundness(P4);

%% TRAINING SET
% Training: A1, A2, A3 (apples) and P1, P2 (pears)
x1_train = [hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2_train = [metric_A1    metric_A2    metric_A3    metric_P1    metric_P2];

% Feature matrix: each column = one sample (2 features)
X_train = [x1_train; x2_train];   % size 2 x 5

% Class labels: apples = 1, pears = 2 (NB-friendly)
T_train = [1 1 1 2 2];

%% ESTIMATE GAUSSIAN NAIVE BAYES PARAMETERS

% Indices of each class
idx_apple = (T_train == 1);
idx_pear  = (T_train == 2);

% Apples: means and variances of each feature
mu_apple  = [mean(x1_train(idx_apple));  mean(x2_train(idx_apple))];
var_apple = [var(x1_train(idx_apple),1); var(x2_train(idx_apple),1)];  % use population var (2nd arg = 1)

% Pears: means and variances of each feature
mu_pear   = [mean(x1_train(idx_pear));   mean(x2_train(idx_pear))];
var_pear  = [var(x1_train(idx_pear),1);  var(x2_train(idx_pear),1)];

% Class priors
N_total   = numel(T_train);
prior_apple = sum(idx_apple) / N_total;
prior_pear  = sum(idx_pear)  / N_total;

% To avoid division by zero in variance
var_apple(var_apple == 0) = 1e-6;
var_pear(var_pear == 0)  = 1e-6;

%% BUILD TEST SET
x1_test = [hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 ...
           hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
x2_test = [metric_A4    metric_A5    metric_A6    metric_A7    ...
           metric_A8    metric_A9    metric_P3    metric_P4];
X_test  = [x1_test; x2_test];   % 2 x N_test

% True labels for test images: apples=1, pears=2
T_test = [1 1 1 1 1 1 2 2];
N_test = numel(T_test);

%% DEFINE HELPER: GAUSSIAN PDF (1D)
gauss_pdf = @(x, mu, varx) (1 ./ sqrt(2*pi*varx)) .* exp( - (x - mu).^2 ./ (2*varx) );

%% RUN NAIVE BAYES CLASSIFIER ON TEST SET
y_test = zeros(1, N_test);

for i = 1:N_test
    x = X_test(:, i);   % x(1)=color, x(2)=roundness
    
    % Likelihoods under each class, assuming feature independence
    % p(x | apple) = p(x1|apple) * p(x2|apple)
    p_x_apple = gauss_pdf(x(1), mu_apple(1), var_apple(1)) * ...
                gauss_pdf(x(2), mu_apple(2), var_apple(2));
    % p(x | pear)
    p_x_pear  = gauss_pdf(x(1), mu_pear(1),  var_pear(1))  * ...
                gauss_pdf(x(2), mu_pear(2),  var_pear(2));
    
    % Posterior scores (proportional)
    score_apple = p_x_apple * prior_apple;
    score_pear  = p_x_pear  * prior_pear;
    
    % Decide class = argmax(score)
    if score_apple >= score_pear
        y_test(i) = 1;   % apple
    else
        y_test(i) = 2;   % pear
    end
end

%% EVALUATE TEST PERFORMANCE
test_errors   = sum(y_test ~= T_test);
test_accuracy = 1 - test_errors / N_test;

fprintf('Naive Bayes - Test misclassified images: %d out of %d\n', test_errors, N_test);
fprintf('Naive Bayes - Test accuracy: %.2f %%\n', 100 * test_accuracy);
