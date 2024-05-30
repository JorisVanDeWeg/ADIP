clear all
close all
clc
%%
% OASIS: dataset with Mild Cognitive Impairment patients and healthy controls
%
% Data are available in small and large format:
%   - subs_05: small, subsampled to 50%
%   - subs_10: large, original size of 100%
% Effect of Age and Total Intracranial Volume (eTIV) regressed out:
%   - dataset          : original data
%       vol contains 4D data
%   - residual_dataset : effects regressed out
%       resid_vol contains 4D data

%load oasis_residual_dataset_subs_10_20150309T105823_97830
%load oasis_residual_dataset_subs_05_20150309T105732_19924
%load oasis_dataset_subs_05_20150309T105732_19924
load oasis_dataset_subs_10_20150309T105823_97830

%%
selected_volume = vol(:,:,:,50);
slice_index = round(size(selected_volume,2) / 2);  % Middle index of the 2nd dimension
slice = squeeze(selected_volume(:,slice_index,:,:));
max = 10 ;
for i= 1:max
    slice_index = round(size(selected_volume,2) * (i/max));  % Middle index of the 2nd dimension
    slice = squeeze(selected_volume(:,slice_index,:,:));
    figure;  % Create a new figure window
    imagesc(slice);  % Display the slice as a scaled image
    axis equal tight;  % Adjust axis for equal spacing and tight fit
    colormap gray;  % Use gray colormap for better visualization
    colorbar;  % Optional: Display a colorbar
end
figure;  % Create a new figure window
imagesc(slice);  % Display the slice as a scaled image
axis equal tight;  % Adjust axis for equal spacing and tight fit
colormap gray;  % Use gray colormap for better visualization
colorbar;  % Optional: Display a colorbar

%%
selected_sub = vol(:,:,:,50);

figure(1); 
montage(selected_sub, []);
title('DICOM Series (Fully Sampled)');


%% Load data A& reshape data
img_data = vol;
% Initialize reshaped data
[nx, ny, nz, num_subjects] = size(img_data);
reshaped_data = zeros(nx * ny * nz, num_subjects);

% Reshape data: stack voxel values from 3D to a row in 2D matrix
for i = 1:num_subjects
    temp = img_data(:,:,:,i);  % Extract the i-th subject's 3D data
    reshaped_data(:, i) = temp(:);  % Convert 3D data to a vector and assign to column
end

% Calculate the mean across the subjects (columns)
mean_vector = mean(reshaped_data, 2);

% Subtract the mean vector from each column (subject)
mean_centered_data = reshaped_data - mean_vector;



%% PCA
% Form the Matrix A
A = mean_centered_data;
% Compute the Covariance Matrix St
St = (A' * A) / num_subjects;
% Solve for the eigenvectors and eigenvalues
[V, Lambda] = eig(St);
% Sort eigenvalues and eigenvectors in descending order
[Lambda_sorted, order] = sort(diag(Lambda), 'descend');
V_sorted = V(:, order);

% Plot eigenvalues (in descending order)
plot(Lambda_sorted)

% Construct the EigenImages
eigenimages = A * V_sorted;
eigenimages = normalize(eigenimages, 1, 'range');


% % Normalize EigenImages
% eigenimages_3d = reshape(eigenimages, [nx,ny,nz,num_subjects]);
% 
% selected_volume = eigenimages_3d(:,:,:,50);
% slice_index = round(size(selected_volume,2) / 2);  % Middle index of the 2nd dimension
% slice = squeeze(selected_volume(:,slice_index,:,:));
% max = 10 ;
% for i= 1:max
%     slice_index = round(size(selected_volume,2) * (i/max));  % Middle index of the 2nd dimension
%     slice = squeeze(selected_volume(:,slice_index,:,:));
%     figure;  % Create a new figure window
%     imagesc(slice);  % Display the slice as a scaled image
%     axis equal tight;  % Adjust axis for equal spacing and tight fit
%     colormap gray;  % Use gray colormap for better visualization
%     colorbar;  % Optional: Display a colorbar
% end
% 
% figure;  % Create a new figure window
% imagesc(slice);  % Display the slice as a scaled image
% axis equal tight;  % Adjust axis for equal spacing and tight fit
% colormap gray;  % Use gray colormap for better visualization
% colorbar;  % Optional: Display a colorbar

% Dimension Reduction with r
r = 181;  % Number of principal components to retain (max 181
P_pca = V_sorted(:, :);  % Principal components matrix

x_pca = A * P_pca;

%% LDA

% Assuming mp is the number of patients and mc is the number of controls
m_p = 60;
m_c = num_subjects - m_p;
patients_data = x_pca(:, 1:m_p);
controls_data = x_pca(:, m_p+1:end);

mean_patients = mean(patients_data, 2);
mean_controls = mean(controls_data, 2);
mean_overall = mean(x_pca, 2);

% Compute within-class scatter matrix Sw
Sw_patients = (patients_data - mean_patients) * (patients_data - mean_patients)';
Sw_controls = (controls_data - mean_controls) * (controls_data - mean_controls)';
Sw = Sw_patients + Sw_controls;

% Compute between-class scatter matrix Sb
Sb_patients = m_p * (mean_patients - mean_overall) * (mean_patients - mean_overall)';
Sb_controls = m_c * (mean_controls - mean_overall) * (mean_controls - mean_overall)';
Sb = Sb_patients + Sb_controls;

% Solve the generalized eigenvalue problem for discriminant directions
[V_lda, D_lda] = eig(Sb, Sw);
[~, order_lda] = sort(diag(D_lda), 'descend');
V_lda_sorted = V_lda(:, order_lda);

% Project the data onto the new LDA space
lda_projected_data = V_lda_sorted(:, 1)'; % Only the first discriminant direction

% Plot the LDA results
figure;
scatter(lda_projected_data * patients_data, zeros(1, m_p), 'r', 'filled');
hold on;
scatter(lda_projected_data * controls_data, ones(1, m_c), 'b', 'filled');
xlabel('LDA Projection');
ylabel('Class');
legend({'Patients', 'Controls'});
title('LDA Projection of Data');
hold off;
