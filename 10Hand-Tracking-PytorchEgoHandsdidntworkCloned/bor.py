import os
import os.path as osp
import random
import scipy
from support_function import *

# Work similar to MyDataset
def join():
    n_elements = -1
    if (n_elements > 2400): n_elements = 450            #???
    entire_folder = int(n_elements / 100)
    folder_list = []
    for element in os.walk(path):
        folder_list.append(element)
        print(f"folder_list contains {len(folder_list)} elements.")
    if folder_list:
        print(folder_list)
#[('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision', ['torchvision', 'docs', 'test', 'references'], ['tox.ini', 'LICENSE', 'hubconf.py', '.travis.yml', 'setup.cfg', '.coveragerc', '.gitignore', 'README.rst', 'MANIFEST.in', '.clang-format', 'setup.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision', ['transforms', 'datasets', 'models', 'csrc', 'ops'], ['__init__.py', 'utils.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\transforms', [], ['functional.py', '__init__.py', 'transforms.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\datasets', [], ['omniglot.py', 'cifar.py', 'svhn.py', 'lsun.py', '__init__.py', 'celeba.py', 'sbd.py', 'utils.py', 'coco.py', 'voc.py', 'flickr.py', 'sbu.py', 'stl10.py', 'mnist.py', 'folder.py', 'imagenet.py', 'phototour.py', 'vision.py', 'semeion.py', 'fakedata.py', 'cityscapes.py', 'caltech.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\models', ['detection', 'segmentation'], ['_utils.py', 'squeezenet.py', 'utils.py', 'densenet.py', '__init__.py', 'resnet.py', 'alexnet.py', 'googlenet.py', 'inception.py', 'mobilenet.py', 'shufflenetv2.py', 'vgg.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\models\\detection', [], ['__init__.py', 'mask_rcnn.py', 'keypoint_rcnn.py', 'backbone_utils.py', 'image_list.py', '_utils.py', 'transform.py', 'rpn.py', 'generalized_rcnn.py', 'roi_heads.py', 'faster_rcnn.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\models\\segmentation', [], ['segmentation.py', '__init__.py', '_utils.py', 'fcn.py', 'deeplabv3.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\csrc', ['cuda', 'cpu'], ['vision.cpp', 'ROIAlign.h', 'ROIPool.h', 'nms.h']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\csrc\\cuda', [], ['ROIPool_cuda.cu', 'vision.h', 'nms_cuda.cu', 'cuda_helpers.h', 'ROIAlign_cuda.cu']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\csrc\\cpu', [], ['vision.h', 'ROIPool_cpu.cpp', 'ROIAlign_cpu.cpp', 'nms_cpu.cpp']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\torchvision\\ops', [], ['poolers.py', '__init__.py', '_utils.py', 'misc.py', 'boxes.py', 'roi_align.py', 'roi_pool.py', 'feature_pyramid_network.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\docs', ['source'], ['Makefile', 'requirements.txt', 'make.bat']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\docs\\source', ['_static'], ['utils.rst', 'datasets.rst', 'conf.py', 'index.rst', 'models.rst', 'transforms.rst']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\docs\\source\\_static', ['css', 'img'], []), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\docs\\source\\_static\\css', [], ['pytorch_theme.css']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\docs\\source\\_static\\img', [], ['pytorch-logo-dark.svg', 'pytorch-logo-flame.png', 'pytorch-logo-flame.svg', 'pytorch-logo-dark.png']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\test', ['assets'], ['preprocess-bench.py', 'test_folder.py', 'sanity_checks.ipynb', 'test_transforms.py', 'test_ops.py', 'smoke_test.py', 'test_models.py', 'test_utils.py', 'test_datasets_utils.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\test\\assets', ['dataset'], ['grace_hopper_517x606.jpg']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\test\\assets\\dataset', ['b', 'a'], []), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\test\\assets\\dataset\\b', [], ['b3.png', 'b1.png', 'b2.png', 'b4.png']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\test\\assets\\dataset\\a', [], ['a2.png', 'a3.png', 'a1.png']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\references', ['detection', 'classification', 'segmentation'], []), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\references\\detection', ['__pycache__'], ['coco_utils.py', 'group_by_aspect_ratio.py', 'engine.py', 'utils.py', 'main.py', 'transforms.py', 'coco_eval.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\references\\detection\\__pycache__', [], ['engine.cpython-311.pyc', 'utils.cpython-311.pyc', 'transforms.cpython-311.pyc', 'coco_utils.cpython-311.pyc', 'coco_eval.cpython-311.pyc']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\references\\classification', [], ['main.py', 'utils.py']), ('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision\\references\\segmentation', [], ['transforms.py', 'main.py', 'utils.py', 'coco_utils.py'])]

        print(folder_list[0])
#('G:\\Meine Ablage\\boris.grillborzer\\10Hand-Tracking-PytorchEgoHandsdidntworkCloned\\RCNN_(egos_hand_dataset)\\vision', ['torchvision', 'docs', 'test', 'references'], ['tox.ini', 'LICENSE', 'hubconf.py', '.travis.yml', 'setup.cfg', '.coveragerc', '.gitignore', 'README.rst', 'MANIFEST.in', '.clang-format', 'setup.py'])

        del folder_list[0]  # Lösche erste Zusammenfassung bzw Vaterpfad 'PennFudanPed'. „Gehe“ nur in die Unterordner und alle darin befindliche Dateien
    else:
        raise ValueError("folder_list is empty. No folders found in the given path.")

    folder_list = random.sample(folder_list, entire_folder + 1)  # shuffle

    elements_list = []
    boxes_list = []
    for folder in folder_list:
        boxes = np.squeeze(scipy.io.loadmat(folder[0] + "/" + folder[2][100])['polygons'])
        print("folder[0] + / + folder[2][100]:", folder[0] + "/" + folder[2][100])
        print(".:", folder[0])

        # Er überspringt den Ordnerzrugriff auf jeden Unterordner mit einfach d\ [2]
        # [G:/user/Projekt/_LABELLEDSAMPLES/polygons.mat] ,[G:/user/Projekt/_LABELLEDSAMPLES/polygons.mat] ,[G:/user/Projekt/_LABELLEDSAMPLES/polygons.mat] , ...
        # Einlesen scipy, # np.squeeze(...) entfernt Dimensionen der Größe 1 aus dem Array, das von ['polygons'] zurückgegeben wird. Dadurch wird das Array in eine kompaktere Form gebracht.
        # aus [[[1]]  wird [1,2,3]
        #      [[2]]               Form vorher (1,3,1)
        #      [[3]]]                  nachher (3,)         #Ist das hilfreich f\uns?
        boxes = getListBoxes(boxes)
        print(boxes)
        print(boxes.shape)
        folder[2].sort()
        print("folder[2]:", folder[2])
        for i, photo_name in zip(range(100), folder[2]):
            elements_list.append(folder[0] + "/" + photo_name)

            boxes_list.append(getBoxes(i, boxes))

    # Shuffle elements
    tmp_index_list = np.linspace(0, len(elements_list) - 1, len(elements_list)).astype(int)
    np.random.shuffle(tmp_index_list)
    elements_list = [elements_list[i] for i in tmp_index_list]
    boxes_list = [boxes_list[j] for j in tmp_index_list]




if __name__ == '__main__':
    path = osp.normpath(osp.join(osp.dirname(__file__), "PennFudanPed"))
    join()
