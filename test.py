import argparse
import torch
from torchvision import transforms

from retinanet import csv_eval, model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    
    csv_annotations_path = 'my_annotations.csv'
    model_path           = 'model_weights.pt'
    class_list_path      = 'my_class_list.csv'
    iou_threshold        = 0.5
    score_threshold      = 0.05
    
    # Create the dataset
    dataset_val = CSVDataset(csv_annotations_path,class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    
    # Create the model
    retinanet=torch.load(model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()

    print(csv_eval.evaluate(dataset_val, retinanet, iou_threshold=iou_threshold, score_threshold=score_threshold))

if __name__ == '__main__':
    main()