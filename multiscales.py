import zarr
from pathlib import Path
from typing import (Union, Iterable)
import numpy as np
from zarr_parallel_processing import defaults


class Multimeta:
    def __init__(self,
                 multimeta = None
                 ):
        if multimeta is None:
            self.multimeta = [{'axes': [],
                               'datasets': [],
                               'name': None,
                               'version': "0.4"
                               }]
        else:
            self.multimeta = multimeta
        self.gr = None

    def __repr__(self):
        return f"Multiscales metadata for indices: {self.axis_order}"

    def from_ngff(self,
                  store: (zarr.Group, zarr.storage.Store, Path, str)
                  ):
        try:
            if isinstance(store, zarr.Group):
                self.gr = store
            else:
                self.gr = zarr.open_group(store, mode = 'a')
            self.multimeta = self.gr.attrs['multiscales']
        except:
            raise Exception(f"The given store does not contain multiscales metadata.")
        return self

    def to_ngff(self,
                  store: (zarr.Group, zarr.storage.Store, Path, str)
                  ):
        try:
            if isinstance(store, zarr.Group):
                self.gr = store
            else:
                self.gr = zarr.open_group(store, mode = 'a')
            self.gr.attrs['multiscales'] = self.multimeta
        except:
            raise Exception(f"The given store is invalid")
        return self

    @property
    def axis_order(self):
        try:
            ret = ''.join([item['name'] for item in self.multimeta[0]['axes']])
        except:
            ret = ''
        return ret

    @property
    def has_axes(self):
        return len(self.axis_order) > 0

    @property
    def is_imglabel(self):
        try:
            func = self.__getattribute__('set_image_label_metadata')
            return func()
        except:
            return False

    @property
    def ndim(self):
        return len(self.axis_order)

    @property
    def nlayers(self):
        return len(self.resolution_paths)

    @property
    def unit_list(self):
        try:
            if self.has_axes:
                l = []
                for item in self.multimeta[0]['axes']:
                    if 'unit' in item.keys():
                        l.append(item['unit'])
                    else:
                        default_unit = defaults.unit_map[item['name']]
                        l.append(default_unit)
                return l
            else:
                return defaults.AxisNotFoundException
        except:
            return defaults.NotMultiscalesException

    @property
    def tag(self):
        return self.multimeta[0]['name']

    def retag(self,
              new_tag: str
              ):
        self.multimeta[0]['name'] = new_tag
        return self

    def _add_axis(self,
                  name: str,
                  unit: str = None,
                  index: int = -1,
                  overwrite: bool = False
                  ):
        if name in self.axis_order:
            if not overwrite:
                raise ValueError(f'{name} axis already exists.')

        def axmake(name, unit):
            if unit is None:
                axis = {'name': name, 'type': defaults.type_map[name]}
            else:
                axis = {'name': name, 'type': defaults.type_map[name], 'unit': unit}
            return axis

        if index == -1:
            index = len(self.multimeta[0]['axes'])
        if not self.has_axes:
            self.multimeta[0]['axes'] = []
            index = 0
        axis = axmake(name, unit)
        if overwrite:
            self.multimeta[0]['axes'][index] = axis
        else:
            self.multimeta[0]['axes'].insert(index, axis)

    def parse_axes(self,
                   axis_order,
                   unit_list: Union[list, tuple] = None,
                   overwrite: bool = None
                   ):
        if len(self.multimeta[0]['axes']) > 0:
            if not overwrite:
                raise ValueError('The current axis metadata is not empty. Cannot overwrite.')
        if unit_list is None:
            unit_list = [None] * len(axis_order)
        elif unit_list == 'default':
            unit_list = [defaults.unit_map[i] for i in axis_order]
        assert len(axis_order) == len(unit_list), 'Unit list and axis order must have the same length.'
        for i, n in enumerate(axis_order):
            self._add_axis(name = n,
                           unit = unit_list[i],
                           index = i,
                           overwrite = overwrite
                           )
        return self

    def rename_paths(self):
        for i, _ in enumerate(self.multimeta[0]['datasets']):
            newkey = str(i)
            oldkey = self.multimeta[0]['datasets'][i]['path']
            self._arrays[newkey] = self._arrays.pop(oldkey)
            self._array_meta_[newkey] = self._array_meta_.pop(oldkey)
            self.multimeta[0]['datasets'][i]['path'] = newkey
        return self

    @property
    def resolution_paths(self):
        try:
            paths = [item['path'] for item in self.multimeta[0]['datasets']]
            return sorted(paths)
        except:
            return []

    def del_dataset(self, path: Union[str, int], hard = False):
        for i, dataset in enumerate(self.multimeta[0]['datasets']):
            if dataset['path'] == str(path):
                del self.multimeta[0]['datasets'][i]
        # if hard:
        #     self.gr.attrs['multiscales'] = self.multimeta
        return

    def add_dataset(self,
                    path: Union[str, int],
                    scale: Iterable[Union[int, float]],
                    translation: Iterable[Union[int, float]] = None,
                    overwrite: bool = False
                    ):
        if not overwrite:
            assert path not in self.resolution_paths, 'Path already exists.'
        assert scale is not None, f"The parameter scale must not be None"
        assert isinstance(scale, (tuple, list))
        transforms = {'scale': scale, 'translation': translation}
        dataset = {'coordinateTransformations': [{f'{key}': list(value), 'type': f'{key}'}
                                                 for key, value in transforms.items()
                                                 if not value is None
                                                 ],
                   'path': str(path)
                   }
        if path in self.resolution_paths:
            idx = self.resolution_paths.index(path)
            self.multimeta[0]['datasets'][idx] = dataset
        else:
            self.multimeta[0]['datasets'].append(dataset)
        args = np.argsort([int(pth) for pth in self.resolution_paths])
        self.multimeta[0]['datasets'] = [self.multimeta[0]['datasets'][i] for i in args]

    @property
    def transformation_types(self):
        transformations = self.multimeta[0]['datasets'][0]['coordinateTransformations']
        return [list(dict.keys())[0] for dict in transformations]

    @property
    def has_translation(self):
        return 'translation' in self.transformation_types

    def get_scale(self,
                  pth: Union[str, int]
                  ):
        pth = str(pth)
        idx = self.resolution_paths.index(pth)
        return self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale']

    def set_scale(self,
                  pth: Union[str, int] = 'auto',
                  scale: Union[tuple, list] = 'auto', #TODO: add dict option
                  hard = False
                  ):
        if isinstance(scale, tuple):
            scale = list(scale)
        elif hasattr(scale, 'tolist'):
            scale = scale.tolist()
        if pth == 'auto':
            pth = self.refpath
        if scale == 'auto':
            pth = self.scales[pth]
        idx = self.resolution_paths.index(pth)
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][0]['scale'] = scale
        if hard:
            self.gr.attrs['multiscales'] = self.multimeta
        return

    def update_scales(self,
                      reference_scale,
                      hard = True
                      ):
        for pth, factor in self.scale_factors.items():
            new_scale = np.multiply(factor, reference_scale)
            self.set_scale(pth, new_scale, hard)
        return self

    def update_unitlist(self, ###TODO: test this
                 unitlist=None,
                 hard=False
                 ):
        if isinstance(unitlist, tuple):
            unitlist = list(unitlist)
        assert isinstance(unitlist, list)
        self.parse_axes(self.axis_order, unitlist, overwrite = True)
        if hard:
            self.gr.attrs['multiscales'] = self.multimeta
        return

    @property
    def scales(self):
        scales = {}
        for pth in self.resolution_paths:
            scl = self.get_scale(pth)
            scales[pth] = scl
        return scales

    def get_translation(self,
                        pth: Union[str, int]
                        ):
        if not self.has_translation: return None
        pth = str(pth)
        idx = self.resolution_paths.index(pth)
        return self.multimeta[0]['datasets'][idx]['coordinateTransformations'][1]['translation']

    def set_translation(self,
                        pth: Union[str, int],
                        translation
                        ):
        if isinstance(translation, np.ndarray):
            translation = translation.tolist()
        idx = self.resolution_paths.index(pth)
        if len(self.multimeta[0]['datasets'][idx]['coordinateTransformations']) < 2:
            self.multimeta[0]['datasets'][idx]['coordinateTransformations'].append({'translation': None, 'type': 'translation'})
        self.multimeta[0]['datasets'][idx]['coordinateTransformations'][1]['translation'] = translation

    @property
    def translations(self):
        translations = {}
        for pth in self.resolution_paths:
            translation = self.get_translation(pth)
            translations[pth] = translation
        return translations

    def del_axis(self,
                 name: str
                 ):
        if name not in self.axis_order:
            raise ValueError(f'The axis "{name}" does not exist.')
        idx = self.axis_order.index(name)
        self.multimeta[0]['axes'].pop(idx)
        for pth in self.resolution_paths:
            scale = self.get_scale(pth)
            scale.pop(idx)
            self.set_scale(pth, scale)
            translation = self.get_translation(pth)
            if translation is not None:
                translation.pop(idx)
                self.set_translation(pth, translation)

    @property
    def label_paths(self):
        try:
            return list(self.labels.keys())
        except:
            return []

    @property
    def has_label_paths(self):
        try:
            return self.labels is not None
        except:
            return False

    @property
    def label_meta(self):
        if self.has_label_paths:
            meta = {'labels': []}
            for name in self.label_paths:
                meta['labels'].append(name)
            return meta
        else:
            return None
