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
        dataset_path = "AG-train.json" if self.split == 'train' else "AG-test.json" 
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

        self.filenames, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships, self.obj_relation_triplets = self.load_graphs(self.ag.dataset, self.split)
        # self.filenames, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships = self.load_graphs2(self.ag.dataset, self.split)

        self.json_category_id_to_contiguous_id = self.class_to_ind
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        
        # print(self.gt_classes[1], self.relationships[1])
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
        # img, target = self.get_groundtruth2(index, img)

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
        obj_relation_triplets = self.obj_relation_triplets[index].copy()
        obj_relations = self.relationships[index].copy() # 3 X N

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
        cache_file = os.path.join('datasets/ag', 'ag_{}_cache.pkl'.format(mode))
        if cache:
            # Load cache if exists
            if os.path.isfile(cache_file):
                with open(cache_file, 'rb') as handle:
                    (filenames, sizes, boxes, gt_classes, relationships, obj_relation_triplets) = pickle.load(handle)
                print('Read imdb from cache at location: {}'.format(cache_file))
                return filenames, sizes, boxes, gt_classes, relationships, obj_relation_triplets
        
        single_anno = 0
        filenames = []; sizes = []; boxes = []; gt_classes = []; relationships = []; obj_relation_triplets = []
        relationship = ['attention_id', 'spatial_id', 'contacting_id']

        for i, imgs in tqdm(enumerate(self.image_index)):
            # Load ag['image'] & ag['annotations']
            _img = self.ag.dataset['images'][i]
            _anno = list(filter(lambda item: item['image_id'] == imgs, self.ag.dataset['annotations']))

            filenames.append(_img['filename'])
            sizes.append(np.array([_img['height'], _img['width']]))

            if len(_anno) == 1 : single_anno += 1
            box_i = []; gt_ci = [1]; rels = np.zeros((3, max(1, len(_anno)))); obj_rel_triplet = []
            #TODO If only one object and no people or only person
            assert _anno != 1 or _anno.pop()['label'] == 'person', 'Only 1 Annotation i.e {}'.format(_anno.pop()['label'])

            for item in _anno:
                # Append all annotations of an image into one array
                assert len(item['bbox']) == 4
                box_i.append(item['bbox'])
                # Append all relationship triplets [0, <catergory_index from gt_ci>, <relationship>]
                
                if item['label'] != 'person':
                    gt_ci.append(item['category_id'])
                    for rel_cat in relationship:
                        assert len(item[rel_cat]) != 0
                        rels[relationship.index(rel_cat), gt_ci.index(item['category_id'])] = item[rel_cat][0]
                        obj_rel_triplet.append([gt_ci.index(1), gt_ci.index(item['category_id']), item[rel_cat][0]])
                       
            assert np.asarray(rels).shape[1] == np.asarray(gt_ci).shape[0]

            box_i = np.array(box_i)
            assert box_i.ndim == 2, 'bbox missing for image_index: {}'.format(_anno)

            boxes.append(box_i)
            gt_classes.append(np.array(gt_ci))
            relationships.append(np.array(rels))
            obj_relation_triplets.append(np.array(obj_rel_triplet))

        sizes = np.stack(sizes, 0)
        
        # Create cache to save time
        if cache:
            with open(cache_file, 'wb') as handle:
                pickle.dump((filenames, sizes, boxes, gt_classes, relationships, obj_relation_triplets), handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Wrote the imdb to cache at location: {}'.format(cache_file))
        
        return filenames, sizes, boxes, gt_classes, relationships, obj_relation_triplets

    def get_groundtruth2(self, index, img):
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


    def load_graphs2(self, annotation, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
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