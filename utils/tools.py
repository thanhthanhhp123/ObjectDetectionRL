
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from utils.config import *
import pandas as pd
import random
import tqdm.notebook as tq

# classes = ['cat', 'cow', 'dog', 'bird', 'car']
classes = ['cat']

def sort_class_extract(datasets):
    """
    Change a whole dataset to seperate datasets corresponding to classes variable
    """
    datasets_per_class = {}
    for j in classes:
        datasets_per_class[j] = {}

    for dataset in datasets:
        for i in tq.tqdm(dataset):
            img, target = i
            obj = target['annotation']['object']
            if isinstance(obj, list):
                classe = target['annotation']['object'][0]["name"]
            else:
                classe = target['annotation']['object']["name"]
            filename = target['annotation']['filename']

            org = {}
            for j in classes:
                org[j] = []
                org[j].append(img)
            
            if isinstance(obj, list):
                for j in range(len(obj)):
                    classe = obj[j]["name"]
                    if classe in classes:
                        org[classe].append([obj[j]["bndbox"], target['annotation']['size']])
            else:
                if classe in classes:
                    org[classe].append([obj["bndbox"], target['annotation']['size']])
            for j in classes:
                if len(org[j]) > 1:
                    try:
                        datasets_per_class[j][filename].append(org[j])
                    except KeyError:
                        datasets_per_class[j][filename] = []
                        datasets_per_class[j][filename].append(org[j])       
    return datasets_per_class


def show_new_bdbox(image, labels, color='r', count=0):
    """
    Imshow an image and corresponding bounding box
    """
    xmin, xmax, ymin, ymax = labels[0],labels[1],labels[2],labels[3]
    fig,ax = plt.subplots(1)
    ax.imshow(image.transpose(0, 2).transpose(0, 1))

    width = xmax-xmin
    height = ymax-ymin
    rect = patches.Rectangle((xmin,ymin),width,height,linewidth=3,edgecolor=color,facecolor='none')
    ax.add_patch(rect)
    ax.set_title("Iteration "+str(count))
    plt.savefig('./temp/'+str(count)+'.png', dpi=100)


def extract(index, loader):
    """
    Extract image and ground truths from a data loader
    ----------
    Argument:
    index              - the index of the img, should be a string of its filename, e.g '00001.jpg'
    loader             - an instance of data loader, 
                         should be a large dict, each key is image filename, value is its information
    ----------
    Return:
    img                - image value, should be (3,224,224)
    ground_truth_boxes - a list of ground truth boxes in this image, 
                         length of this list should equal to how many objects there are in this image
    """
    extracted = loader[index]
    img = extracted[0][0]
    ground_truth_boxes =[]
    for ex in extracted[0][1:]:
        bndbox = ex[0]
        size = ex[1]
        xmin = ( float(bndbox['xmin']) /  float(size['width']) ) * 224
        xmax = ( float(bndbox['xmax']) /  float(size['width']) ) * 224

        ymin = ( float(bndbox['ymin']) /  float(size['height']) ) * 224
        ymax = ( float(bndbox['ymax']) /  float(size['height']) ) * 224

        ground_truth_boxes.append([xmin, xmax, ymin, ymax])
    return img, ground_truth_boxes


def voc_ap(rec, prec, voc2007=True):
    if voc2007:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def intersection_over_union(box1, box2):
        """
        Compute IoU value over two bounding boxes
        Each box is represented by four elements vector: (left, right, top, bottom)
        Origin point of image system is on the top left
        """
        box1_left, box1_right, box1_top, box1_bottom = box1
        box2_left, box2_right, box2_top, box2_bottom = box2
        
        inter_top = max(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = min(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_bottom - inter_top)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou

def prec_rec_compute(bounding_boxes, gt_boxes, ovthresh):
    
    nd = 0
    for each in gt_boxes:
        nd += len(each)
    npos = nd
    
    ious = np.zeros(nd)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    d = 0

    for index in range(len(bounding_boxes)):
        for gtbox in gt_boxes[index]:
            best_iou = 0
            for bdbox in bounding_boxes[index]:
                iou = intersection_over_union(gtbox,bdbox)
                if iou > best_iou:
                    best_iou = iou
                    
            if best_iou > ovthresh:
                tp[d] = 1.0
            else:            
                fp[d] = 1.0
                
            ious[d] = best_iou
            
            d += 1
            
    sort_idx = np.argsort(-ious)
    fp = fp[sort_idx]
    tp = tp[sort_idx]
    ious = ious[sort_idx]
    
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return prec, rec


def compute_ap_and_recall(all_bdbox, all_gt, ovthresh):
    prec, rec = prec_rec_compute(all_bdbox, all_gt, ovthresh)
    ap = voc_ap(rec, prec, False)
    return ap, rec[-1]


def eval_stats_at_threshold(all_bdbox, all_gt, thresholds=[0.4, 0.5, 0.6]):
    stats = {}
    for ovthresh in thresholds:
        ap, recall = compute_ap_and_recall(all_bdbox, all_gt, ovthresh)
        stats[ovthresh] = {'ap': ap, 'recall': recall}
    stats_df = pd.DataFrame.from_records(stats)*100
    return stats_df


class ReplayMemory(object):
    """
    A replay memory object to do expereience replay.
    Each sample is 
    """
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)