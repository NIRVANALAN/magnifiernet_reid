# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .cuhk03_d import CUHK03_D
from .cuhk03_l import CUHK03_L
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .veri import VeRi
from .dataset_loader import ImageDataset
from .mt_dataset_loader import MTImageDataset
from .import_DukeMTMCAttribute import import_DukeMTMCAttribute_binary
from .import_Market1501Attribute import import_Market1501Attribute_binary


__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'cuhk03_d': CUHK03_D,
    'cuhk03_l': CUHK03_L,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'veri': VeRi,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
