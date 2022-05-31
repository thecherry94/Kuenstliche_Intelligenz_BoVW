import utils

def main():
    annotations = utils.annotate_dataset('../mvtec_anomaly_detection_data', debug=True)
    utils.create_annotation_files(annotations)


if __name__ == 'main':
    main()