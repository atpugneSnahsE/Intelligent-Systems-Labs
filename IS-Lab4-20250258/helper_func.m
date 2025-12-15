function pozymiai = helper_func(pavadinimas,pvz_eiluciu_sk, mode)
%% Feature extraction for handwritten digits (Robust Version)

%% Step 1: Read and Preprocess
I = imread(pavadinimas);
if ndims(I) == 3
    I = rgb2gray(I);
end

% Adaptive binarization with noise removal
BW = imbinarize(I, 'adaptive', 'ForegroundPolarity','dark', 'Sensitivity', 0.5);
BW = imcomplement(BW);

% Morphological cleaning
se = strel('disk', 1);
BW = imopen(BW, se);        % Remove small noise
BW = imclose(BW, se);       % Connect broken strokes
BW = bwareaopen(BW, 50);    % Remove very small blobs
BW = imfill(BW, 'holes');

%% Step 2: Connected Components
CC = bwconncomp(BW, 8);
stats = regionprops(CC, 'BoundingBox', 'Centroid', 'Area', 'EulerNumber', 'Image');

% Filter noise by area (keep objects larger than threshold)
stats = stats([stats.Area] > 150); 
num_objects = numel(stats);

if num_objects == 0
    warning('No objects detected!');
    pozymiai = {};
    return;
end

%% Step 3: Robust Sorting (Row-wise)
centroids = cat(1, stats.Centroid);
sorted_stats = [];

if strcmp(mode, 'train')
    % TRAINING: Sort into strict grid (Rows then Columns)
    % 1. Sort primarily by Y to find rows
    [~, y_idx] = sort(centroids(:,2));
    temp_stats = stats(y_idx);
    temp_cents = centroids(y_idx, :);
    
    % 2. Group into rows based on Y-distance threshold
    row_indices = {};
    current_row = 1;
    row_indices{1} = 1;
    
    for i = 2:length(temp_cents)
        % If Y difference is small, it's the same row
        if abs(temp_cents(i,2) - temp_cents(i-1,2)) < 35 % Threshold depends on image text size
            row_indices{current_row} = [row_indices{current_row}, i];
        else
            current_row = current_row + 1;
            row_indices{current_row} = i;
        end
    end
    
    % 3. Sort X (Left-to-Right) within each row
    for r = 1:length(row_indices)
        idx_in_row = row_indices{r};
        row_blobs = temp_stats(idx_in_row);
        row_cents = temp_cents(idx_in_row, :);
        
        [~, x_sort] = sort(row_cents(:,1));
        sorted_stats = [sorted_stats; row_blobs(x_sort)];
    end
    stats = sorted_stats;
else
    % TESTING: Simple Left-to-Right sort
    [~, xidx] = sort(centroids(:,1));
    stats = stats(xidx);
end

%% Step 4: Feature Extraction
num_processed = numel(stats);
pozymiai = cell(1, num_processed);

for k = 1:num_processed
    % Extract image locally
    digit = stats(k).Image;
    
    % Resize to standard grid for density calculation
    digit = imresize(digit, [70 50]);
    
    feature = zeros(49, 1);
    idx = 1;
    
    % A. Block density (7x5 grid) -> 35 features
    h = 70/7; w = 50/5;
    for i = 1:7
        for j = 1:5
            r1 = round((i-1)*h + 1); r2 = round(i*h);
            c1 = round((j-1)*w + 1); c2 = round(j*w);
            block = digit(r1:r2, c1:c2);
            feature(idx) = sum(block(:)) / numel(block);
            idx = idx + 1;
        end
    end
    
    % B. Horizontal Projection -> 7 features
    for i = 1:7
        r1 = round((i-1)*h + 1); r2 = round(i*h);
        strip = digit(r1:r2, :);
        feature(idx) = sum(strip(:)) / numel(strip);
        idx = idx + 1;
    end
    
    % C. Vertical Projection -> 5 features
    for j = 1:5
        c1 = round((j-1)*w + 1); c2 = round(j*w);
        strip = digit(:, c1:c2);
        feature(idx) = sum(strip(:)) / numel(strip);
        idx = idx + 1;
    end
    
    % D. Structural Features
    % Normalize Euler number (usually 1, 0, or -1 for digits)
    feature(idx) = (stats(k).EulerNumber + 1) / 3; 
    idx = idx + 1;
    
    % Aspect Ratio (normalized roughly to 0-1 range)
    bb = stats(k).BoundingBox;
    ar = bb(3)/bb(4);
    feature(idx) = ar / 2; % Divide by 2 to keep it roughly < 1
    
    pozymiai{k} = feature;
end
end