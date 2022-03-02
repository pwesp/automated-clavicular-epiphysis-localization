import torch
from   torchvision import transforms

import numpy as np
import pandas as pd
from   pathlib import Path
from   retinanet import csv_predict, model
from   retinanet.dataloader import CSVDataset, Resizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

# Directories containing slices of CT scans (stored as jpgs)
ct_scan_dirs = ['data/ct_scan_0', 'data/ct_scan_1', 'data/ct_scan_2']

# Specify trained model weights
model_path = 'model_weights.pt'

# Specify class list
class_list_path = 'my_class_list.csv'

# Write annotations for validation set images
for ct_scan_dir in ct_scan_dirs:

    annot_strings = []

    for ct_scan in Path(ct_scan_dir).iterdir():
        if str(ct_scan.name).startswith('ae_') and ct_scan.is_file():

            annot_string = str(ct_scan) + ',,,,,'
            annot_strings.append(annot_string)

    annot_strings.sort()
            
    with open(ct_scan_dir+"/validation_annotations.csv", "w") as csv_file:
        for annot_string in annot_strings:
            csv_file.write(annot_string+'\n')

def model_inference(class_list_path, ct_scan_dirs, model_path):

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
    
    for ct_scan_dir in ct_scan_dirs:
        
        print('\nApply model to images in: {:s}'.format(ct_scan_dir))
        
        csv_annotations_path = ct_scan_dir + '/validation_annotations.csv'
        
        dataset_val = CSVDataset(csv_annotations_path, class_list_path, transform=transforms.Compose([Normalizer(), Resizer()]))

        # Load annotations
        print('Load annotations from: {:s}'.format(csv_annotations_path))
        df_annots = pd.read_csv(csv_annotations_path, names=['image_path','x1','y1','x2','y2','class_name'])

        print('Load images and make predictions...')
        all_detections = csv_predict.predict(dataset_val, retinanet, score_threshold=0.0)
        n_detections   = len(all_detections)
        print('Number of detections: {:d}'.format(n_detections))

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
        ct_scan_id  = df_annots['image_path'][0].split('/')[1]
        
        predictions_path = 'results/' + ct_scan_id + '/' + ct_scan_id + '_predictions.csv'
        predictions_path = Path(predictions_path)

        if not predictions_path.parent.is_dir():
            predictions_path.parent.mkdir(parents=True)

        print('Save results: {:s}'.format(str(predictions_path)))
        with open(str(predictions_path), "w") as csv_file:
            for pred_string in predictions_string:
                csv_file.write(pred_string)

model_inference(class_list_path, ct_scan_dirs, model_path)