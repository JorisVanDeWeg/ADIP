function [vol, stats]= getData(option)
    switch option
        case 'A'
            load oasis_dataset_subs_10_20150309T105823_97830
            fprintf('Original dataset will be used \n Shape:')
            disp(size(vol))
        case 'B'
            load oasis_dataset_subs_05_20150309T105732_19924
            fprintf('Subsampled dataset will be used \n Shape:')
            disp(size(vol))
        case 'C'
            load oasis_residual_dataset_subs_10_20150309T105823_97830
            fprintf('Residual dataset will be used \n Shape:')
            vol = resid_vol;
            disp(size(vol))
        case 'D'
            load oasis_residual_dataset_subs_05_20150309T105732_19924
            fprintf('Subsampled residual dataset will be used \n Shape:')
            vol = resid_vol;
            disp(size(vol))
        otherwise
            error("Wrong dataset option was given")
    end
    fprintf('\n')
end