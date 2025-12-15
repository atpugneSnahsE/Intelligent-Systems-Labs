close all;
clc

%% Step 1: Load training data
train_image = 'train_data.png'; 
% Note: Ensure your training image has roughly 10 digits per row
features_train = helper_func(train_image, 10, 'train');
P = cell2mat(features_train);

% Check data integrity
if isempty(P)
    error('No training data extracted. Check image path/content.');
end

num_samples = size(P, 2);
fprintf('Training samples extracted: %d\n', num_samples);

%% Step 2: Create Target Matrix
T = zeros(10, num_samples);
for i = 1:num_samples
    digit_val = mod(i-1, 10) + 1; % Maps 1->1 ... 10->0
    if digit_val == 10
        target_row = 10; % Row 10 represents digit '0'
    else
        target_row = digit_val; % Rows 1-9 represent '1'-'9'
    end
    T(target_row, i) = 1;
end

%% Step 3: RBF Training
spread = 30;  
goal_error = 0.001;
max_neurons = 10; 
net = newrb(P, T, goal_error, spread, max_neurons);

%% Step 4: Test Data Extraction
test_image = 'test_2.png';
features_test = helper_func(test_image, 1, 'test');
P_test = cell2mat(features_test);

if isempty(P_test)
    error('No test data detected.');
end

%% Step 5: Simulation and Result
Y = sim(net, P_test);

% Find the row with maximum activation
[max_val, detected_indices] = max(Y);

% Map indices back to digits
detected_digits = detected_indices;
detected_digits(detected_indices == 10) = 0;

%% Step 6: Visualization
figure('Color', 'w');
subplot(2,1,1);
imshow(imread(test_image));
title('Original Test Image');

subplot(2,1,2);
axis off;
result_str = num2str(detected_digits);
% Remove extra spaces for cleaner display
result_str = result_str(result_str ~= ' '); 

text(0.5, 0.5, result_str, ...
    'FontSize', 32, ...
    'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'BackgroundColor', [0.9 0.9 0.9], ...
    'EdgeColor', 'k');
title(['Recognized Sequence (Spread: ' num2str(spread) ')']);

fprintf('Detected: %s\n', result_str);