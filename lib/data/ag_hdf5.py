import os
from collections import defaultdict
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from lib.utils.box import bbox_overlaps


class ag_hdf5(Dataset):
    def __init__(self, cfg, split='train', transforms=None, num_im=-1, filter_duplicate_rels=True):
        self.split = split
        self.root = cfg.AGDATASET.PATH
        self.transforms = transforms
        assert os.path.exists(self.root), "Cannot find the Action Genome dataset at {}".format(self.root)
        self.roidb_file = os.path.join(self.root, 'annotations', 'COCO')
        self.image_file = os.path.join(self.root, 'frames') 
        dataset_path = "action_genome_train_v3graph_v2.json" if self.split == 'train' else "action_genome_val_v3graph_v2.json" 
        self.ag = COCO(os.path.join(self.roidb_file, dataset_path))
        self.image_index = list(sorted(self.ag.imgs.keys()))
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'

        self.ind_to_classes = []; self.class_to_ind = dict(); self.ind_to_predicates = []; self.predicate_to_ind = dict()

        for cat in self.ag.dataset['categories']:
            if cat['supercategory'] == 'object':
                self.class_to_ind[cat['name']] = cat['id']
                self.ind_to_classes.append(cat['name'])
            else: 
                self.predicate_to_ind[cat['name']] = cat['id']
                self.ind_to_predicates.append(cat['name'])

        self.ind_to_classes.insert(0, '__background__'); self.class_to_ind['__background__'] = 0
        self.ind_to_predicates.insert(0, '__background__'); self.predicate_to_ind['__background__'] = 0

        self.filenames, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships = self.load_graphs(self.ag.dataset, self.split)
        
        self.json_category_id_to_contiguous_id = self.class_to_ind
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}

        # img, target, index = self.__getitem__(1)
        # print(type(img))
        # exit()

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        fauxcoco = self.ag
        fauxcoco.createIndex()
        return fauxcoco

    def __getitem__(self, index):
        # Own coco file
        img = Image.open(os.path.join(self.image_file, self.filenames[index])).copy()
        img, target = self.get_groundtruth(index, img)

        return img, target, index

    def __len__(self):
        return len(self.image_index)

    def get_img_info(self, img_id):
        w, h = self.im_sizes[img_id, :]
        return {"height": h, "width": w}

    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_classes[class_id]

    def get_groundtruth(self, index, img):
        width, height = self.im_sizes[index, :]
        # get object bounding boxes, labels and relations

        obj_boxes = self.gt_boxes[index].copy()
        obj_labels = self.gt_classes[index].copy()
        obj_relation_triplets = self.relationships[index].copy()

        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))

        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        target_raw = BoxList(obj_boxes, (width, height), mode="xyxy")
        img, target = self.transforms(img, target_raw)
        target.add_field("labels", torch.from_numpy(obj_labels))
        target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
        target = target.clip_to_image(remove_empty=False)
        return img, target

    def load_graphs(self, annotation, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                    filter_non_overlap=False, cache=True):
        # Check mode
        if mode not in ('train', 'val', 'test'):
            raise ValueError('{} invalid'.format(mode))
        cache_file = os.path.join('datasets/ag', 'ag_graph_{}_cache.pkl'.format(mode))
        if cache:
            # Load cache if exists
            if os.path.isfile(cache_file):
                with open(cache_file, 'rb') as handle:
                    (filenames, sizes, boxes, gt_classes, relationships) = pickle.load(handle)
                print('Read imdb from cache at location: {}'.format(cache_file))
                return filenames, sizes, boxes, gt_classes, relationships
        
        sizes = []; boxes = []; gt_classes = []; relationships = []

        for i, imgs in tqdm(enumerate(self.image_index)):
            # Load ag['image'] & ag['annotations']
            _img = self.ag.dataset['images'][i]
            _anno = list(filter(lambda item: item['image_id'] == imgs, self.ag.dataset['annotations']))

            filenames.append(_img['filename'])
            sizes.append(np.array([_img['height'], _img['width']]))
            
            box_i = []; rels = []; gt_ci = [1]
            #TODO If only one object and no people or only person
            
            for item in _anno:
                # Append all annotations of an image into one array
                assert len(item['bbox']) == 4
                box_i.append(item['bbox'])
                # Append all relationship triplets [0, <catergory_index from gt_ci>, <relationship>]
                if item['label'] != 'person':
                    gt_ci.append(item['category_id'])
                    rels.append([gt_ci.index(1), gt_ci.index(item['category_id']), item['contacting_id'].pop()])
                    #TODO Same for other relations

                assert np.asarray(rels).shape[0] <= np.asarray(gt_ci).shape[0]

            box_i = np.array(box_i)
            assert box_i.ndim == 2, 'bbox missing for image_index: {}'.format(_anno)

            boxes.append(box_i)
            gt_classes.append(np.array(gt_ci))
            relationships.append(np.array(rels))

        sizes = np.stack(sizes, 0)
        
        # Create cache to save time
        if cache:
            with open(cache_file, 'wb') as handle:
                pickle.dump((filenames, sizes, boxes, gt_classes, relationships), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Wrote the imdb to cache at location: {}'.format(cache_file))
        
        return filenames, sizes, boxes, gt_classes, relationships

# [{'id': 372891829902578154, 'label': 'table', 'bbox': [222.10317460317458, 143.829365079365, 479.88095238095235, 244.9404761904761], 
# 'area': 26064.19753086419, 'iscrowd': 0, 'category_id': 31, 'image_id': 0, 'attention_relationship': ['unsure'], 'attention_id': [38], 
# 'spatial_relationship': ['in_front_of'], 'spatial_id': [41], 'contacting_relationship': ['not_contacting'], 'contacting_id': [53]}]
# {'id': 372891834197545450, 'label': 'chair', 'bbox': [56.34126984126985, 179.16666666666663, 249.11904761904762, 269.7355687782746],
#  'area': 17459.671684848872, 'iscrowd': 0, 'category_id': 7, 'image_id': 0, 'attention_relationship': ['not_looking_at'], 'attention_id': [37],
#  'spatial_relationship': ['beneath', 'behind'], 'spatial_id': [40, 42], 'contacting_relationship': ['sitting_on', 'leaning_on'], 'contacting_id': [51, 55]}
# {'id': 372933345056461290, 'label': 'person', 'bbox': [24.297740936279297, 71.44395446777344, 259.23602294921875, 268.202880859375], 
# 'category_id': 0, 'area': 46226.20413715328, 'image_id': 0, 'iscrowd': 0}


    # if 'attention_id' in item:
    #     gt_ci.extend(item['attention_id'])
    # if 'spatial_id' in item:
    #     gt_ci.extend(item['spatial_id'])
    # if 'contacting_id' in item:
    #     gt_ci.extend(item['contacting_id'])
    
    # Append all relationship triplets [0. <relationship>, <category_id>]
    # if item['label'] != 'person':
    #     rels = []
    #     for j in range(len(item['attention_relationship'])):
    #         rels.append([0, item['category_id'], item['attention_id'][j]])
    #     for j in range(len(item['contacting_relationship'])):
    #         rels.append([0, item['category_id'], item['contacting_id'][j]])
    #     for j in range(len(item['spatial_relationship'])):
    #         rels.append([0, item['category_id'], item['spatial_id'][j]])

    #     relationships.append(np.array(rels))