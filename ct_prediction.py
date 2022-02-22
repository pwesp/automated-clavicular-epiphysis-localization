import torch
from   torchvision import transforms

import numpy as np
import pandas as pd
from   pathlib import Path
from   retinanet import csv_predict, model
from   retinanet.dataloader import CSVDataset, Resizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

# Load list of scans in validation set
valid_scans = np.loadtxt('data/clavicula_sets/test_scans.csv', delimiter=",", dtype=str)

# Get image directories containing slices of the validation set (stored as jpgs)
image_dirs = []

for study in valid_scans:
    patient   = study.split('_')[0]
    image_dir = '/data/public/age_estimation/jpg/ae_{:s}/ae_{:s}'.format(patient, study)
    image_dirs.append(image_dir)
    
# Write annotations for validation set images
for image_dir in image_dirs:

    annot_strings = []

    for image in Path(image_dir).iterdir():
        if str(image.name).startswith('ae_') and image.is_file():

            annot_string = str(image) + ',,,,,'
            annot_strings.append(annot_string)

    with open(image_dir+"/annots.csv", "w") as csv_file:
        for annot_string in annot_strings:
            csv_file.write(annot_string+'\n')

class_list_path      = 'clavicula_class_list.csv'
model_path           = 'model_final_v2.pt'

def predict_ct(class_list_path, image_dirs, model_path):

    # Create the model
    retinanet=torch.load(model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    
    for image_dir in image_dirs:
        
        print('\nPredict image_dir: {:s}'.format(image_dir))
        
        csv_annotations_path = image_dir + '/annots.csv'
        
        dataset_val = CSVDataset(csv_annotations_path, class_list_path, transform=transforms.Compose([Normalizer(), Resizer()]))

        # Load annotations
        print('Load annotations: {:s}'.format(csv_annotations_path))
        df_annots = pd.read_csv(csv_annotations_path, names=['image_path','x1','y1','x2','y2','class_name'])  

        all_detections = csv_predict.predict(dataset_val, retinanet, score_threshold=0.0)
        n_detections   = len(all_detections)
        print('number of detections: {:d}'.format(n_detections))

        predictions_string = []
        class_col          = 0

        for i, detection in enumerate(range(n_detections)):

            # Check if detection array is empty or not
            is_detection = all_detections[i][class_col].shape[0]

            if is_detection:
                #print('detection #{:d}: {}'.format(i, all_detections[i][class_col][0]))
                x1         = str(all_detections[i][class_col][0][0])
                y1         = str(all_detections[i][class_col][0][1])
                x2         = str(all_detections[i][class_col][0][2])
                y2         = str(all_detections[i][class_col][0][3])
                score      = str(all_detections[i][class_col][0][4])
            else:
                #print('no detection!')
                x1         = str(0)
                y1         = str(0)
                x2         = str(1)
                y2         = str(1)
                score      = str(0)

            image_path = df_annots['image_path'][i]

            # Construct string
            pred_string = image_path + ',' + x1 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + score + '\n'
            predictions_string.append(pred_string)

        # Save results
        study_id = df_annots['image_path'][0].split('age_estimation/jpg/')[-1].split('/')[0]
        scan_id  = df_annots['image_path'][0].split('age_estimation/jpg/')[-1].split('/')[1]
        
        predictions_path = 'test_results_v2/' + study_id + '/' + scan_id + '.csv'
        predictions_path = Path(predictions_path)

        if not predictions_path.parent.is_dir():
            predictions_path.parent.mkdir(parents=True)

        print('Save results: {:s}'.format(str(predictions_path)))
        with open(str(predictions_path), "w") as csv_file:
            for pred_string in predictions_string:
                csv_file.write(pred_string)

predict_ct(class_list_path, image_dirs, model_path)