from model import config
import sys
import os
import torch
import cv2
import numpy

# load our object detector, set it evaluation mode, and label

if len(sys.argv) < 2:
    print("Please enter the path to the model to be evaluated")
    sys.exit(1)

model_path = sys.argv[1]

print(f"**** loading object detector at {model_path}...")
model = torch.load(model_path).to(config.DEVICE)
model.eval()
print(f"**** object detector loaded")

results_labels = dict()

for mode, csv_file in [['train', config.TRAIN_PATH],
                       ['validation', config.VAL_PATH],
                       ['test', config.TEST_PATH],]:
    data = []
    assert(csv_file.endswith('.csv'))

    print(f"Evaluating {mode} set...")
    # loop over CSV file rows (filename, startX, startY, endX, endY, label)
    for row in open(csv_file).read().strip().split("\n"):
        # TODO: read bounding box annotations
        filename, startX, startY, endX, endY, label = row.split(',')
        filename = os.path.join(config.IMAGES_PATH, label, filename)
        # TODO: add bounding box annotations here
        data.append((filename, startX, startY, endX, endY, label))

    print(f"Evaluating {len(data)} samples...")

    # Store all results as well as per class results
    results_labels[mode] = dict()
    results_labels[mode]['all'] = []
    for label_str in config.LABELS:
        results_labels[mode][label_str] = []

    # loop over the images that we'll be testing using our bounding box
    # regression model
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
        # load the image, copy it, swap its colors channels, resize it, and
        # bring its channel dimension forward
        image = cv2.imread(filename)
        display = image.copy()
        h, w = display.shape[:2]

        # convert image to PyTorch tensor, normalize it, upload it to the
        # current device, and add a batch dimension
        image = config.TRANSFORMS(image).to(config.DEVICE)
        image = image.unsqueeze(0)

        # predict the bounding box of the object along with the class label
        label_predictions, bbox_predictions = model(image)

        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        # TODO: denormalize bounding box from (0,1)x(0,1) to (0,w)x(0,h)
        pred_box = bbox_predictions.squeeze(0).cpu().detach()
        pred_start_x = int(pred_box[0] * w)
        pred_start_y = int(pred_box[1] * h)
        pred_end_x = int(pred_box[2] * w)
        pred_end_y = int(pred_box[3] * h)

        # Compare to gt data
        results_labels[mode]['all'].append(label == gt_label)
        results_labels[mode][gt_label].append(label == gt_label)

        # TODO: compute cumulated bounding box metrics
        gt_sx, gt_sy = int(gt_start_x), int(gt_start_y)
        gt_ex, gt_ey = int(gt_end_x),   int(gt_end_y)
        inter_x1 = max(pred_start_x, gt_sx)
        inter_y1 = max(pred_start_y, gt_sy)
        inter_x2 = min(pred_end_x,   gt_ex)
        inter_y2 = min(pred_end_y,   gt_ey)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        pred_area  = max(0, pred_end_x - pred_start_x) * max(0, pred_end_y - pred_start_y)
        gt_area    = max(0, gt_ex - gt_sx)             * max(0, gt_ey - gt_sy)
        union_area = pred_area + gt_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        results_iou[mode]['all'].append(iou)
        results_iou[mode][gt_label].append(iou)

        if label != gt_label:
            print(f"\tFailure at {filename}")


# Compute per dataset accuracy
for mode in ['train', 'validation', 'test']:
    print(f'\n*** {mode} set accuracy')
    print(f"\tMean accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_labels[mode]['all']))}")
    # TODO: display bounding box metrics
    all_iou = numpy.array(results_iou[mode]['all'])
    print(f"\tMean IoU for all labels: {numpy.mean(all_iou):.4f}")

    for label_str in config.LABELS:
        print(f'\n\tMean accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_labels[mode][label_str]))}')
        print(f'\t\t {numpy.sum(results_labels[mode][label_str])} over '
              f'{len(results_labels[mode][label_str])} samples')
        # TODO: display bounding box metrics
        class_iou = numpy.array(results_iou[mode][label_str])
        print(f'\t\tMean IoU: {numpy.mean(class_iou):.4f}')
        print(f'\t\tIoU >= 0.5: {numpy.mean(class_iou >= 0.5):.4f}')

