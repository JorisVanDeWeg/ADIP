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

load oasis_residual_dataset_subs_10_20150309T105823_97830
%load oasis_residual_dataset_subs_05_20150309T105732_19924
%load oasis_dataset_subs_05_20150309T105732_19924
% load oasis_dataset_subs_10_20150309T105823_97830

%%
% Extract eTIV and CDR values
eTIV = stats.eTIV;
CDR = stats.CDR;

% Create a figure
figure;
hold on;

% Plot subjects with CDR = 0 (no Alzheimer's)
no_alzheimer = CDR == 0;
scatter(find(no_alzheimer), eTIV(no_alzheimer), 'b', 'DisplayName', 'No Alzheimer''s (CDR=0)');

% Plot subjects with CDR = 0.5 (mild Alzheimer's)
mild_alzheimer = CDR == 0.5;
scatter(find(mild_alzheimer), eTIV(mild_alzheimer), 'r', 'DisplayName', 'Mild Alzheimer''s (CDR=0.5)');

% Customize the plot
xlabel('Subject Index');
ylabel('eTIV');
title('eTIV Values Colored by CDR');
legend;
grid on;
hold off;

%%
% Extract Age and CDR values
Age = stats.Age;
CDR = stats.CDR;

% Create a figure
figure;
hold on;

% Plot subjects with CDR = 0 (no Alzheimer's)
no_alzheimer = CDR == 0;
scatter(find(no_alzheimer), Age(no_alzheimer), 'b', 'DisplayName', 'No Alzheimer''s (CDR=0)');

% Plot subjects with CDR = 0.5 (mild Alzheimer's)
mild_alzheimer = CDR == 0.5;
scatter(find(mild_alzheimer), Age(mild_alzheimer), 'r', 'DisplayName', 'Mild Alzheimer''s (CDR=0.5)');

% Customize the plot
xlabel('Subject Index');
ylabel('Age');
title('Age Values Colored by CDR');
legend;
grid on;
hold off;



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

% Load the selected volume
selected_volume = resid_vol(:,:,:,50);

% Determine the middle index for each dimension
mid_x = round(size(selected_volume, 1) / 2);  % Middle index of the 1st dimension
mid_y = round(size(selected_volume, 2) / 2);  % Middle index of the 2nd dimension
mid_z = round(size(selected_volume, 3) / 2);  % Middle index of the 3rd dimension

% Extract slices for transverse (axial), sagittal, and coronal views
transverse_slice = squeeze(selected_volume(:, :, mid_z));
sagittal_slice = squeeze(selected_volume(mid_x, :, :));
coronal_slice = squeeze(selected_volume(:, mid_y, :));

% Plot the transverse (axial) slice
figure;
imagesc(transverse_slice);
axis equal tight;
colormap gray;
colorbar;
title('Transverse Slice');

% Plot the sagittal slice
figure;
imagesc(squeeze(sagittal_slice)');
axis equal tight;
colormap gray;
colorbar;
title('Sagittal Slice');

% Plot the coronal slice
figure;
imagesc(squeeze(coronal_slice)');
axis equal tight;
colormap gray;
colorbar;
title('Coronal Slice');
