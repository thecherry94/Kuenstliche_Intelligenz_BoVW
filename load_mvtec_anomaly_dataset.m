function [] = load_mvtec_anomaly_dataset(directory)
    contents = dir(directory);
    dirFlags = [contents.isdir];
    subFolders = contents(dirFlags);

    datasets = [];
    for i = 3 : length(subFolders)
        
    end
end

