function [predicted_train_labels, predicted_test_labels] = classify_data(lda_train_proj, lda_test_proj, labels_train, labels_test, mean_lda_patients, mean_lda_controls, classifier)
    % Classification function
    if strcmp(classifier, 'nn')
        predicted_train_labels = zeros(size(labels_train));
        for j = 1:length(lda_train_proj)
            if abs(lda_train_proj(j) - mean_lda_patients) < abs(lda_train_proj(j) - mean_lda_controls)
                predicted_train_labels(j) = 1;
            else
                predicted_train_labels(j) = 0;
            end
        end
        
        predicted_test_labels = zeros(size(labels_test));
        for j = 1:length(lda_test_proj)
            if abs(lda_test_proj(j) - mean_lda_patients) < abs(lda_test_proj(j) - mean_lda_controls)
                predicted_test_labels(j) = 1;
            else
                predicted_test_labels(j) = 0;
            end
        end
        
    elseif strcmp(classifier, 'svm')
        % Train SVM classifier
        svm_model = fitcsvm(lda_train_proj', labels_train, 'KernelFunction', 'linear');
        
        % Predict training and test labels
        predicted_train_labels = predict(svm_model, lda_train_proj');
        predicted_test_labels = predict(svm_model, lda_test_proj');
        
    else
        error('Unknown classifier type');
    end
end