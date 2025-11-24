% Classification using perceptron

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

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;-1;-1];

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% calculate weighted sum with randomly generated parameters
v1 = w1 * P(1,1) + w2 * P(2,1) + b;
% calculate current output of the perceptron 
if v1 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e1 = T(1) - y;

% calculate wieghted sum with randomly generated parameters
v2 = w1 * P(1,2) + w2 * P(2,2) + b;
% calculate current output of the perceptron 
if v2 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e2 = T(2) - y;

% calculate wieghted sum with randomly generated parameters
v3 = w1 * P(1,3) + w2 * P(2,3) + b;
% calculate current output of the perceptron 
if v3 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e3 = T(3) - y;

% calculate wieghted sum with randomly generated parameters
v4 = w1 * P(1,4) + w2 * P(2,4) + b;
% calculate current output of the perceptron 
if v4 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e4 = T(4) - y;

% calculate wieghted sum with randomly generated parameters
v5 = w1 * P(1,5) + w2 * P(2,5) + b;
% calculate current output of the perceptron 
if v5 > 0
	y = 1;
else
	y = -1;
end
% calculate the error
e5 = T(5) - y;

% calculate the total error for these 5 inputs 
e = abs(e1) + abs(e2) + abs(e3) + abs(e4) + abs(e5);%total error

eta = 0.1;
e_list = zeros(1,5); %storage

while e ~= 0

    for i = 1:5
        v = w1 * x1(i) + w2 * x2(i) + b;
        y = 1 * (v > 0) - 1 * (v <= 0);
        e_current = T(i) - y;

        w1 = w1 + eta * e_current * x1(i);
        w2 = w2 + eta * e_current * x2(i);
        b  = b  + eta * e_current;
    end

    for i = 1:5
        v = w1 * x1(i) + w2 * x2(i) + b;
        y = 1 * (v > 0) - 1 * (v <= 0);
        e_list(i) = T(i) - y;
    end

    e = sum(abs(e_list));
    fprintf('Total error = %d\n', e);
end

fprintf('Final weights: w1=%.4f, w2=%.4f, b=%.4f\n', w1, w2, b);

%% TEST SET (images NOT used for training)
x1_test = [hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 ...
           hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
x2_test = [metric_A4    metric_A5    metric_A6    metric_A7    ...
           metric_A8    metric_A9    metric_P3    metric_P4];
P_test  = [x1_test; x2_test];

% Desired labels for test images: apples=+1, pears=-1
T_test = [1 1 1 1 1 1 -1 -1];

%% 4) RUN CLASSIFIER ON TEST SET AND COMPUTE ERROR
N_test = numel(T_test);
y_test = zeros(1,N_test);

for i = 1:N_test
    v = w1 * P_test(1,i) + w2 * P_test(2,i) + b;
    y_test(i) = 1 * (v > 0) - 1 * (v <= 0);
end

% classification error (number and percentage of misclassified images)
test_errors = sum(y_test ~= T_test);
test_accuracy = 1 - test_errors / N_test;

fprintf('Test misclassified images: %d out of %d\n', test_errors, N_test);
fprintf('Test accuracy: %.2f %%\n', 100 * test_accuracy);