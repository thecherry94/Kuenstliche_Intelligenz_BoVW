function dataset = load_mvtec_anomaly_dataset(directory)
    contents = dir(directory);
    dirFlags = [contents.isdir];
    subFolders = contents(dirFlags);
    for i = 3 : length(subFolders)
        subdir = subFolders(i);
        fprintf('loading defect type "%s" (%d/%d)', subdir.name, i - 2, length(subFolders));
        dataset(end+1) = load_mvtec_anomaly_dataset_subdirectory(subdir);
    end
end

