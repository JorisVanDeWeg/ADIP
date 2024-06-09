clear all
close all
clc

% Parameters and running variables
dataset_options = 'A':'D';
r_values = 1:1:50;
num_r = length(r_values);
k = 5;  % Number of folds
l = 10; % Number of repetitions for robustness

% Loop through each dataset option
for dataset_idx = 1:length(dataset_options)
    dataset_option = dataset_options(dataset_idx);
    
    % Data Load of OASIS with getData function
    [imgData, stats] = getData(dataset_option); % [3D Images x num Subjects, labels data]
    
    % Initialize the labeling list
    labels_CDR = zeros(size(stats.CDR));
    % Update the list: set to 1 if CDR element is not zero
    labels_CDR(stats.CDR ~= 0) = 1;

    % Reshape data
    [nx, ny, nz, num_subjects] = size(imgData);
    reshaped_data = zeros(nx * ny * nz, num_subjects);

    % Reshape data: stack voxel values from 3D to a row in 2D matrix
    for i = 1:num_subjects
        temp = imgData(:,:,:,i);  % Extract the i-th subject's 3D data
        reshaped_data(:, i) = temp(:);  % Convert 3D data to a vector and assign to column
    end

    % Calculate the mean across the subjects 
    mean_vector = mean(reshaped_data, 2);
    % Subtract the mean vector from each column (subject)
    mean_centered_data = reshaped_data - mean_vector;

    % Initialize arrays to store errors for l runs
    train_errors = zeros(num_r, k, l);
    test_errors = zeros(num_r, k, l);

    TP_train = zeros(num_r, k, l);
    TN_train = zeros(num_r, k, l);
    FP_train = zeros(num_r, k, l);
    FN_train = zeros(num_r, k, l);
    TP_test = zeros(num_r, k, l);
    TN_test = zeros(num_r, k, l);
    FP_test = zeros(num_r, k, l);
    FN_test = zeros(num_r, k, l);


    % Start timing
    tic;

    % Repeat the cross-validation process l times
    for run_idx = 1:l
        % Cross-validation setup
        indices = crossvalind('Kfold', labels_CDR, k);

        % For range r_values
        for r_idx = 1:num_r
            r = r_values(r_idx);
            
            % For every fold k
            for i = 1:k
                test = (indices == i);
                train = ~test;
                
                % Split data into train and test data
                A_train = mean_centered_data(:, train);
                A_test = mean_centered_data(:, test);
                labels_train = labels_CDR(train);
                labels_test = labels_CDR(test);
                
                % PCA on training data
                St_train = (A_train' * A_train) / sum(train);
                [V, Lambda] = eig(St_train);
                [Lambda_sorted, order] = sort(diag(Lambda), 'descend');
                V_sorted = V(:, order);
                
                % Dimension Reduction with r
                P_pca_train = A_train * V_sorted(:, 1:r);
                
                % Project training and test data onto the PCA space
                x_pca_train = P_pca_train' * A_train;
                x_pca_test = P_pca_train' * A_test;
                
                % LDA on training data
                m_p = sum(labels_train == 1); 
                m_c = sum(labels_train == 0);
                
                % Extract indices and find data for patients and controls separately
                patients_data_train = x_pca_train(:, labels_train == 1);
                controls_data_train = x_pca_train(:, labels_train == 0);
                
                % Find the mean
                mean_patients = mean(patients_data_train, 2);
                mean_controls = mean(controls_data_train, 2);
                mean_overall = mean(x_pca_train, 2);
                
                % Compute within-class scatter matrix Sw
                Sw_patients = (patients_data_train - mean_patients) * (patients_data_train - mean_patients)';
                Sw_controls = (controls_data_train - mean_controls) * (controls_data_train - mean_controls)';
                Sw = (1/size(x_pca_train, 2)) .* (Sw_patients + Sw_controls);
                
                % Compute between-class scatter matrix Sb
                Sb_patients = m_p * (mean_patients - mean_overall) * (mean_patients - mean_overall)';
                Sb_controls = m_c * (mean_controls - mean_overall) * (mean_controls - mean_overall)';
                Sb = (1/size(x_pca_train, 2)) .* (Sb_patients + Sb_controls);
                
                % Solve the generalized eigenvalue problem for discriminant directions
                [V_lda, Lambda_lda] = eig(Sb, Sw);
                [~, order_lda] = sort(diag(Lambda_lda), 'descend');
                V_lda_sorted = V_lda(:, order_lda);
                
                % Project the training and test data onto the new LDA space
                q = V_lda_sorted(:, 1); % Only the first discriminant direction
                
                % LDA projection
                lda_train_proj = q' * x_pca_train;
                lda_test_proj = q' * x_pca_test;
                
                % Classification using nearest mean classifier
                mean_lda_patients = mean(lda_train_proj(labels_train == 1));
                mean_lda_controls = mean(lda_train_proj(labels_train == 0));
                
                method = 'nn';
                % Function for classification
                [predicted_train_labels, predicted_test_labels] = classify_data(lda_train_proj, lda_test_proj, labels_train, labels_test, mean_lda_patients, mean_lda_controls, method);
                  
                % Compute error for this fold
                train_errors(r_idx, i, run_idx) = sum(predicted_train_labels ~= labels_train) / length(labels_train);
                test_errors(r_idx, i, run_idx) = sum(predicted_test_labels ~= labels_test) / length(labels_test);

                % Compute TP, TN, FP, FN for training set
                TP_train(r_idx, i, run_idx) = sum((predicted_train_labels == 1) & (labels_train == 1));
                TN_train(r_idx, i, run_idx) = sum((predicted_train_labels == 0) & (labels_train == 0));
                FP_train(r_idx, i, run_idx) = sum((predicted_train_labels == 1) & (labels_train == 0));
                FN_train(r_idx, i, run_idx) = sum((predicted_train_labels == 0) & (labels_train == 1));
                
                % Compute TP, TN, FP, FN for testing set
                TP_test(r_idx, i, run_idx) = sum((predicted_test_labels == 1) & (labels_test == 1));
                TN_test(r_idx, i, run_idx) = sum((predicted_test_labels == 0) & (labels_test == 0));
                FP_test(r_idx, i, run_idx) = sum((predicted_test_labels == 1) & (labels_test == 0));
                FN_test(r_idx, i, run_idx) = sum((predicted_test_labels == 0) & (labels_test == 1));
            end
        end
    end

    % Stop timing
    elapsed_time = toc;
    fprintf('Elapsed time for dataset %s: %.2f seconds\n', dataset_option, elapsed_time);
    fprintf('Average time per run for dataset %s: %.2f seconds\n', dataset_option, elapsed_time/ l);

    % Compute average errors and standard deviations over l runs for each value of r
    average_train_errors = mean(mean(train_errors, 2), 3);
    average_test_errors = mean(mean(test_errors, 2), 3);
    std_train_errors = std(mean(train_errors, 2), 0, 3);
    std_test_errors = std(mean(test_errors, 2), 0, 3);

    % Compute average TP, TN, FP, FN over l runs for each value of r
    average_TP_train = mean(mean(TP_train, 2), 3);
    average_TN_train = mean(mean(TN_train, 2), 3);
    average_FP_train = mean(mean(FP_train, 2), 3);
    average_FN_train = mean(mean(FN_train, 2), 3);

    average_TP_test = mean(mean(TP_test, 2), 3);
    average_TN_test = mean(mean(TN_test, 2), 3);
    average_FP_test = mean(mean(FP_test, 2), 3);
    average_FN_test = mean(mean(FN_test, 2), 3);

    ltr = length(labels_train);
    lte = length(labels_test);

    % Create directory for saving the results
    result_dir = fullfile('results', sprintf('Dataset_%s', dataset_option));
    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end



    % Plot the results
    figure;
    plot(r_values, average_train_errors, 'DisplayName', 'Training Error');
    hold on;
    plot(r_values, average_test_errors, 'DisplayName', 'Testing Error');
    xlabel('Number of Principal Components (r)');
    ylabel('Classification Error');
    % title(sprintf('OASIS %d\nTraining and Testing Errors for Different Values of r', dataset_option));
    legend;
    grid on;
    hold off;
    saveas(gcf, fullfile(result_dir, 'Classification_Error.png'));



    % Plot the results with error bars
    figure;
    errorbar(r_values, average_train_errors, std_train_errors, 'DisplayName', 'Training Error');
    hold on;
    errorbar(r_values, average_test_errors, std_test_errors, 'DisplayName', 'Testing Error');
    xlabel('Number of Principal Components (r)');
    ylabel('Classification Error');
    % title(sprintf('OASIS %d\nTraining and Testing Errors for Different Values of r', dataset_option));
    legend;
    grid on;
    hold off;
    saveas(gcf, fullfile(result_dir, 'Classification_Error_With_Error_Bars.png'));



    % Plot TP, FP, FN, and TN results
    % Subplot 1: True Positives
    figure('Position', [100, 100, 1200, 800]);
    subplot(2, 2, 1);
    plot(r_values, average_TP_train/ltr, 'DisplayName', 'Train TP');
    hold on;
    plot(r_values, average_TP_test/lte, 'DisplayName', 'Test TP');
    xlabel('Number of Principal Components (r)');
    ylabel('True Positives Rate');
    title('True Positives');
    legend;
    grid on;

    % Subplot 2: False Positives
    subplot(2, 2, 2);
    plot(r_values, average_FP_train/ltr, 'DisplayName', 'Train FP');
    hold on;
    plot(r_values, average_FP_test/lte, 'DisplayName', 'Test FP');
    xlabel('Number of Principal Components (r)');
    ylabel('False Positives Rate');
    title('False Positives');
    legend;
    grid on;

    % Subplot 3: False Negatives
    subplot(2, 2, 3);
    plot(r_values, average_FN_train/ltr, 'DisplayName', 'Train FN');
    hold on;
    plot(r_values, average_FN_test/lte,'DisplayName', 'Test FN');
    xlabel('Number of Principal Components (r)');
    ylabel('False Negatives Rate');
    title('False Negatives');
    legend;
    grid on;

    % Subplot 4: True Negatives
    subplot(2, 2, 4);
    plot(r_values, average_TN_train/ltr, 'DisplayName', 'Train TN');
    hold on;
    plot(r_values, average_TN_test/lte, 'DisplayName', 'Test TN');
    xlabel('Number of Principal Components (r)');
    ylabel('True Negatives Rate');
    title('True Negatives');
    legend;
    grid on;

    % sgtitle(sprintf('Performance Metrics for Different Values of r - Dataset %d', dataset_option));
    saveas(gcf, fullfile(result_dir, 'Performance_Metrics.png'));
end
