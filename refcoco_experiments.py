import sys
refer_path = "D:\\Datasets\\refer-master\\refer-master"
img_path = refer_path+"\\data\\images\\mscoco\\images\\train2014\\"
sys.path.append(refer_path)
from refer import REFER
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pprint import pprint
from PIL import Image
# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union
    
    
data_root = refer_path + '/data'  # contains refclef, refcoco, refcoco+, refcocog and images
dataset = 'refcoco+'
splitBy = 'unc'
refer = REFER(data_root, dataset, splitBy)


ref_ids = refer.getRefIds()
print(len(ref_ids))
print (len(refer.Imgs))
print (len(refer.imgToRefs))

ref_ids = refer.getRefIds(split='train')
print ('There are %s training referred objects.' % len(ref_ids))



for ref_id in ref_ids:
    ref = refer.loadRefs(ref_id)[0]
    image_id = refer.getImgIds(ref_id)[0]
    if len(ref['sentences']) < 2:
        continue

    pprint(ref)
    print ('LSeg label %s.' % refer.Cats[ref['category_id']]) #pass to lseg
    #args.label_src = refer.Cats[ref['category_id']] + ",others"
    

    image = Image.open(img_path+refer.loadImgs(image_id)[0]['file_name'])
    print(type(image))
    print(image)
    plt.imshow(image)
    #extract bounding boxes
    #if only 1 bounding box, just return that one
    print('pass to clip:')
    for sentence in ref['sentences']:
        print (sentence['sent'])
    #run clip with 3 prompts, use voting to determine which to return. only return 1 bbox for each prompts
    
    plt.figure()
    refer.showRef(ref, seg_box='box')
    plt.show()