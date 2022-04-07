function [name, ground_truth, test, train] = load_mvtec_anomaly_dataset_subdirectory(directory)
    name = directory.name;
    contents = dir(directory);
    dirFlags = [contents.isdir];
    subFolders = contents(dirFlags);

    for i = 3 : length(subFolders)
        subdir = subFolders(i);
        
    end
end

