"""
data module docstring.

The data module is a wrapper for multiple datasets. It hides low-level details, i.e. reading image, getting labels, and etc,
and provides high-level functionalities to `datasets.py`

To use a dataset, you must define following functions:
    - get_[dataset]_cls_names, which returns the names of all classes in the [dataset]
    - get_[dataset]_data, which returns the all the images and labels of the [ dataset ]
    - get_[dataset]_cls_data_getter, which returns a callable function that returns the images and labels of a given class every time you call it, since class-incremental learning splits all the classes in the [dataset] into different tasks. So, a function that returns the data of a class is necessary
    - get_[dataset]_task_data_getter, which returns a callable function that returns the images and labels of a given task every time you call it. It basically calls [dataset]_cls_data_getter repeatedly to load data of all classes in the task
    - get_[dataset]_tasks, which splits the classes of the [dataset] into tasks
    - [dataset]Dataset, which is the implementation of pytorch.utils.data.Dataset

Check `cifar100.py` for more detail

Author: Shihong Wang
Date: 2024/5/5
"""
