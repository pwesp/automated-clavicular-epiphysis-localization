import numpy as np
import pandas as pd
from   pathlib import Path



def get_annotated_slices(annotation_file):
    df_annotations               = pd.read_csv(annotation_file, names=['image_path','x1','y1','x2','y2','class_name'])
    df_annotated_slices          = df_annotations.dropna().copy(deep=True)
    ct_ids                       = [x.split('/')[1] for x in df_annotated_slices['image_path'].to_list()]
    df_annotated_slices['ct_id'] = ct_ids
    return df_annotated_slices



def get_detections(results_dir, df_annotated_slices):

    # List all predictions
    results_dir = Path(results_dir)

    retina_net_detections = []
    study_ids             = []
    for i in results_dir.iterdir():

        if str(i.name).startswith('ct_') and i.is_dir():
            for j in i.iterdir():

                study_id = j.stem.split('_predictions')[0]

                if str(j.name).startswith('ct_') and str(j.name).endswith('.csv') and j.is_file():
                    retina_net_detections.append(str(j))
                    study_ids.append(str(study_id))
                    
    # Filter predictions for which there is a ground truth available
    detections_data = np.array([retina_net_detections, study_ids]).swapaxes(0,1)
    df_detections   = pd.DataFrame(detections_data, columns=['retina_net_detections', 'ct_id'])

    detection_with_groundtruth = np.where(df_detections['ct_id'].isin(df_annotated_slices['ct_id']))[0].tolist()
    df_detections              = df_detections.iloc[detection_with_groundtruth]
    df_detections              = df_detections.sort_values(by='ct_id')
    df_detections              = df_detections.reset_index(drop=True)
    return df_detections



def get_ground_truth_slices(scan, df_annotated_slices):
    
    # Load ground truth slice info
    df_ct_scan  = df_annotated_slices.iloc[np.where(df_annotated_slices['ct_id'].isin([scan]))[0].tolist()]
    image_paths = df_ct_scan['image_path'].to_list()
    
    true_slices = [x.split('_s_')[1].split('.jpg')[0] for x in image_paths]
    true_slices = np.asarray(true_slices, dtype=np.int32)
    true_slices = np.sort(true_slices)
    
    return true_slices



def get_ground_truth_bboxes(scan, df_labelled_slices):
    
    # Load ground truth slice info
    df_ct_scan  = df_labelled_slices.iloc[np.where(df_labelled_slices['ct_id'].isin([scan]))[0].tolist()]
    
    bboxes = []
    for row in df_ct_scan.iterrows():
        
        row = row[1]
        
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
    
        bbox = [x1, y1, x2, y2]
        bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes)
    return bboxes



def get_scores_and_bboxes(detections_file):
    """
    Load bounding box predictions for an entire CT scan from a csv file and return the slice scores
    
    Parameters
    ----------
    detections_file : string
        Path to csv file with detections
    """
    header = ['image_path','x1','y1','x2','y2','score']
    
    # Load prediction for a full CT scan
    df_detections = pd.read_csv(detections_file, names=header)
    
    # Add column with slice ID to data frame
    image_paths               = df_detections['image_path'].to_list()
    slice_id                  = [x.split('_s_')[1].split('.jpg')[0] for x in image_paths]
    slice_id                  = np.asarray(slice_id, dtype=np.int32)
    df_detections['slice_id'] = slice_id
    
    # Sort data frame according to slice ID
    df_detections              = df_detections.sort_values(by='slice_id')
    df_detections              = df_detections.reset_index(drop=True)
    
    # Extract bounding box coordinates and slice scores
    x1     = df_detections['x1'].to_numpy()
    y1     = df_detections['y1'].to_numpy()
    x2     = df_detections['x2'].to_numpy()
    y2     = df_detections['y2'].to_numpy()
    scores = df_detections['score'].to_numpy()
    
    # Calculate bounding box parameters
    #w    = x2 - x1
    #h    = y2 - y1
    bbox = [x1, y1, x2, y2]
    bbox = np.asarray(bbox).swapaxes(0,1)
    
    return scores, bbox