import numpy as np
import pandas as pd
from   pathlib import Path
from   utils.metrics import IoU
from   utils.sets import difference, intersection, union



def evaluate_object_detection(df_detections, df_annotated_slices, print_confusion_matrix=False, print_scans=False):
    # Loop over predictions
    print('Found predictions for n={:d} CT scans'.format(df_detections.shape[0]))

    ct_scans         = []
    true_slices      = []
    detected_slices  = []
    IoUs             = []

    n_slices = 0
    n_annots = 0
    n_tp     = 0
    n_tn     = 0
    n_fp     = 0
    n_fn     = 0

    scores_in_annots = []
    scores_in_fps    = []
    scores_in_tns    = []
    IoUs_in_annots   = []

    for i, detection_ct in enumerate(df_detections['retina_net_detections']):

        score_threshold = 0.05
        IoU_thresholds  = 0.5

        scan                    = detection_ct.split('/')[1]
        true_slices_ct          = get_ground_truth_slices(scan, df_annotated_slices)
        true_bbox_ct            = get_ground_truth_bboxes(scan, df_annotated_slices)
        scores_ct, pred_bbox_ct = get_scores_and_bboxes(detection_ct)

        # Find slices with a detection
        slices_ct          = np.arange(scores_ct.shape[0])
        detected_slices_ct = np.empty([0], dtype=np.int64)
        IoUs_detection_ct  = np.empty([0], dtype=np.float64)
        IoUs_true_ct       = np.empty([0], dtype=np.float64)
        for s, score, pred_bbox in zip(slices_ct, scores_ct, pred_bbox_ct):

            # Check if detection score is above threshold
            if score > score_threshold:
                detected_slices_ct = np.append(detected_slices_ct, s)

                # Calculate IoU
                IoU_slice = 0

                # Check if the detection is in an annotated slice
                if s in true_slices_ct:
                    true_bbox = true_bbox_ct[np.where(true_slices_ct==s)][0]
                    IoU_slice = IoU(pred_bbox, true_bbox)

                IoUs_detection_ct = np.append(IoUs_detection_ct, IoU_slice)

            # Check if the slice is in an annotated slice
            if s in true_slices_ct:

                # Calculate IoU
                IoU_slice = 0

                # Check if detection score is above threshold
                if score > score_threshold:
                    true_bbox = true_bbox_ct[np.where(true_slices_ct==s)][0]
                    IoU_slice = IoU(pred_bbox, true_bbox)

                IoUs_true_ct = np.append(IoUs_true_ct, IoU_slice)

        ct_scans.append(scan)
        true_slices.append(true_slices_ct)
        detected_slices.append(detected_slices_ct)
        IoUs.append(IoUs_true_ct)

        if print_scans:
            print('\n************************************')
            print('Scan: {:s}'.format(scan))
            print('\tTrue slices:     {}'.format(true_slices_ct))    
            print('\tDetected slices: {}'.format(detected_slices_ct))
            print('\tIoUs:            {}'.format(IoUs_true_ct))

        # Confusion matrix: Calculate TP, TN, FP, FN
        n_slices_ct     = slices_ct.shape[0]
        n_annots_ct     = true_slices_ct.shape[0]
        n_detections_ct = detected_slices_ct.shape[0]
        tp_ct =  intersection(true_slices_ct, detected_slices_ct)
        tn_ct =  slices_ct[~np.isin(slices_ct, union(true_slices_ct, detected_slices_ct))]
        fp_ct =  difference(detected_slices_ct, true_slices_ct)
        fn_ct =  difference(true_slices_ct, detected_slices_ct)

        n_slices += n_slices_ct
        n_annots += n_annots_ct
        n_tp += tp_ct.shape[0]
        n_tn += tn_ct.shape[0]
        n_fp += fp_ct.shape[0]
        n_fn += fn_ct.shape[0]

        if print_confusion_matrix:
            print('\nConfusion matrix:')
            print('\tn_slices     = {:d}'.format(n_slices_ct))
            print('\tn_annots     = {:d}'.format(n_annots_ct))
            print('\t# TP         = {:d}/{:d} ({:.2f} %)'.format(tp_ct.shape[0], n_annots_ct,             100*(float(tp_ct.shape[0])/float(n_annots_ct))))
            print('\t# TN         = {:d}/{:d} ({:.2f} %)'.format(tn_ct.shape[0], n_slices_ct-n_annots_ct, 100*(float(tn_ct.shape[0])/float(n_slices_ct-n_annots_ct))))
            print('\t# FP         = {:d}/{:d} ({:.2f} %)'.format(fp_ct.shape[0], n_slices_ct-n_annots_ct, 100*(float(fp_ct.shape[0])/float(n_slices_ct-n_annots_ct))))
            print('\t# FN         = {:d}/{:d} ({:.2f} %)'.format(fn_ct.shape[0], n_annots_ct,             100*(float(fn_ct.shape[0])/float(n_annots_ct))))

        # Classification score
        scores_in_annots_ct  = scores_ct[true_slices_ct]
        scores_in_fp_ct      = scores_ct[fp_ct]
        scores_in_tn_ct      = scores_ct[tn_ct]

        scores_in_annots.extend(scores_in_annots_ct)
        scores_in_fps.extend(scores_in_fp_ct)
        scores_in_tns.extend(scores_in_tn_ct)

        mean_score_in_annots = np.mean(scores_in_annots_ct)

        # IoU: Average IoU in true slice
        IoUs_in_annots.extend(IoUs_true_ct)

    print('\n************************************')
    print('Summary:')
    print('\tn_scans (total)  = {:d}'.format(df_detections.shape[0]))
    print('\tn_slices (total) = {:d}'.format(n_slices))
    print('\tn_annots (total) = {:d}'.format(n_annots))
    print('\t# TP             = {:d}/{:d} ({:.1f} %)'.format(n_tp, n_annots,          100*(float(n_tp)/float(n_annots))))
    print('\t# TN             = {:d}/{:d} ({:.1f} %)'.format(n_tn, n_slices-n_annots, 100*(float(n_tn)/float(n_slices-n_annots))))
    print('\t# FP             = {:d}/{:d} ({:.1f} %)'.format(n_fp, n_slices-n_annots, 100*(float(n_fp)/float(n_slices-n_annots))))
    print('\t# FN             = {:d}/{:d} ({:.1f} %)'.format(n_fn, n_annots,          100*(float(n_fn)/float(n_annots))))
    print('\n\tMean score in annots = {:.2f} [{:.2f},{:.2f}]'.format(np.median(scores_in_annots), np.quantile(scores_in_annots, q=0.25), np.quantile(scores_in_annots, q=0.75)))
    if n_fp > 1:
        print('\tMean score in fps    = {:.2f} [{:.2f},{:.2f}]'.format(np.median(scores_in_fps),    np.quantile(scores_in_fps, q=0.25),    np.quantile(scores_in_fps, q=0.75)))
    if n_tn > 1:
        print('\tMean score in tns    = {:.2f} [{:.2f},{:.2f}]'.format(np.median(scores_in_tns),    np.quantile(scores_in_tns, q=0.25),    np.quantile(scores_in_tns, q=0.75)))
    print('\tMean IoU in annots   = {:.2f} [{:.2f},{:.2f}]'.format(np.median(IoUs_in_annots), np.quantile(IoUs_in_annots, q=0.25), np.quantile(IoUs_in_annots, q=0.75)))
    print('\tMean IoU in detect   = {:.2f} [{:.2f},{:.2f}]'.format(np.median(IoUs_detection_ct), np.quantile(IoUs_detection_ct, q=0.25), np.quantile(IoUs_detection_ct, q=0.75)))

    

def find_nearest(array,value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
    
 
    
def evaluate_soi_localization(df_detections, df_annotated_slices, print_scans=False):
    # Loop over predictions
    print('Found predictions for n={:d} CT scans'.format(df_detections.shape[0]))

    scans            = []
    scores           = []
    bbox             = []
    localized_slices = []
    true_slices      = []
    hits             = []
    xy_dists_tp      = []
    xy_dists_fp      = []

    tp_det = 0
    fp_det = 0
    fn_det = 0

    for detection_ct in df_detections['retina_net_detections']:

        score_threshold = 0.05

        scan               = detection_ct.split('/')[1]
        true_slices_ct     = get_ground_truth_slices(scan, df_annotated_slices)
        true_bbox_ct       = get_ground_truth_bboxes(scan, df_annotated_slices)
        scores_ct, bbox_ct = get_scores_and_bboxes(detection_ct)

        # Check if there is a detection with a classification score above the threshold
        if np.amax(scores_ct) > score_threshold:
            # Pick the slice(s) with the highest classification score. Typically, this will be a single slice
            loc_slice = np.where(scores_ct==np.amax(scores_ct))[0]

            if loc_slice.shape[0] > 1:
                print("WARNING: Localized more than 1 slice!")
                break

            # Check if one or more of the picked slices is in the list of true slices
            slice_overlap = set(loc_slice).intersection(set(true_slices_ct))
            slice_overlap = list(slice_overlap)

            # Store results
            if len(slice_overlap) > 0:
                hits.append(1)
                tp_det +=1

                # xy-plane distance
                true_bbox = true_bbox_ct[np.where(true_slices_ct==loc_slice)][0]
                loc_bbox  = bbox_ct[loc_slice][0]

                true_x = true_bbox[2] - true_bbox[0]
                true_y = true_bbox[3] - true_bbox[1]

                loc_x  = loc_bbox[2] - loc_bbox[0]
                loc_y  = loc_bbox[3] - loc_bbox[1]

                loc_xy_distance = np.sqrt( (loc_x-true_x)**2 + (loc_y-true_y)**2 )
                xy_dists_tp.append(loc_xy_distance)

            else:
                hits.append(0)
                fp_det += 1

                # Find nearest true slice
                nearest_true_slice = find_nearest(true_slices_ct, loc_slice)

                # xy-plane distance
                true_bbox = true_bbox_ct[np.where(true_slices_ct==nearest_true_slice)][0]
                loc_bbox  = bbox_ct[loc_slice][0]

                true_x = true_bbox[2] - true_bbox[0]
                true_y = true_bbox[3] - true_bbox[1]

                loc_x  = loc_bbox[2] - loc_bbox[0]
                loc_y  = loc_bbox[3] - loc_bbox[1]

                loc_xy_distance = np.sqrt( (loc_x-true_x)**2 + (loc_y-true_y)**2 )
                xy_dists_fp.append(loc_xy_distance)

        else:
            loc_slice = np.empty(shape=(0))
            hits.append(0)
            fn_det += 1
        if print_scans:
            print('\n************************************')    
            print('Scan: {:s}'.format(scan))
            print('\tTrue slices:      {}'.format(true_slices_ct))
            print('\tLocalized slices: {}'.format(loc_slice))

        scans.append(scan)
        scores.append(scores_ct)
        bbox.append(np.asarray(bbox_ct))
        localized_slices.append(loc_slice)
        true_slices.append(true_slices_ct)

    hits = np.asarray(hits)
    print('\n************************************')
    print('Summary:')
    print('\nConfusion matrix:')
    print('\t#TP = {:d}'.format(tp_det))
    print('\t#FP = {:d}'.format(fp_det))
    print('\t#FN = {:d}'.format(fn_det))

    n_tot = hits.shape[0]
    n_0   = np.where(hits==0)[0].shape[0]
    n_1   = np.where(hits==1)[0].shape[0]
    accuracy = (float(n_tot-n_0) / float(n_tot)) * 100
    print('\nAccuracy = {:d}/{:d} ({:.2f} %)'.format(n_1, n_tot, accuracy))

    # Localization distance
    print('\nLocalization distance:')
    if tp_det>0:
        print('\tAverage localization distance in xy-plane for TPs: {:.1f} +/- {:.1f} pxl (n={:d})'.format(np.mean(xy_dists_tp), np.std(xy_dists_tp), len(xy_dists_tp)))

    if fp_det>0:
        print('\tAverage localization distance in xy-plane for FPs: {:.1f} +/- {:.1f} pxl (n={:d})'.format(np.mean(xy_dists_fp), np.std(xy_dists_fp), len(xy_dists_fp)))
    


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