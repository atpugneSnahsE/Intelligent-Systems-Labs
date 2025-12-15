close all;
clc;

%% HANDWRITTEN DIGIT RECOGNITION – MLP
%% Step 1: Load training data & Extract Features
train_image = 'train_data.png'; 

% Using the SAME feature extraction function
features_train = helper_func(train_image, 10, 'train');
P = cell2mat(features_train);

if isempty(P)
    error('No training data extracted. Check image path.');
end

num_samples = size(P, 2);
fprintf('Training samples extracted: %d\n', num_samples);

%% Step 2: Create Target Matrix
T = zeros(10, num_samples);
for i = 1:num_samples
    digit_val = mod(i-1, 10) + 1; 
    if digit_val == 10
        target_row = 10; 
    else
        target_row = digit_val;
    end
    T(target_row, i) = 1;
end

%% Step 3: Configure MLP Network
hiddenLayerSize = 20; 

% Create Pattern Recognition Network
net = patternnet(hiddenLayerSize);

% Setup Training Parameters
net.trainFcn = 'trainscg';        % Scaled Conjugate Gradient (Fast)
net.performFcn = 'crossentropy';  % Better loss function for classification
net.trainParam.epochs = 1000;     % Max epochs
net.trainParam.goal = 1e-5;       % Target error
net.trainParam.min_grad = 1e-6;   % Minimum gradient
net.trainParam.showWindow = true; % Show training GUI

% Data Division (70% Train, 15% Validation, 15% Test)
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;

%% Step 4: Train the MLP
fprintf('Training MLP Network (%d hidden neurons)...\n', hiddenLayerSize);

% Fix random seed for reproducibility (Optional)
rng(1); 

[net, tr] = train(net, P, T);

% Plot performance (Optional)
% figure; plotperform(tr);

%% Step 5: Test Data Extraction
test_image = 'test_2.png';
features_test = helper_func(test_image, 1, 'test');
P_test = cell2mat(features_test);

if isempty(P_test)
    error('No test data detected.');
end

%% Step 6: Simulation and Interpretation
Y = net(P_test); % Or sim(net, P_test)

% Find the row with maximum probability
[max_val, detected_indices] = max(Y);

% Map indices back to digits
detected_digits = detected_indices;
detected_digits(detected_indices == 10) = 0;

%% Step 7: Visualization
figure('Color', 'w');
subplot(2,1,1);
imshow(imread(test_image));
title('Original Test Image');

subplot(2,1,2);
axis off;
result_str = num2str(detected_digits);
result_str = result_str(result_str ~= ' '); % Remove spaces

text(0.5, 0.5, result_str, ...
    'FontSize', 32, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.9 0.9 1.0], ...
    'EdgeColor', 'b');
title(['MLP Recognition Result (' num2str(hiddenLayerSize) ' hidden neurons)']);

fprintf('Detected Sequence: %s\n', result_str);