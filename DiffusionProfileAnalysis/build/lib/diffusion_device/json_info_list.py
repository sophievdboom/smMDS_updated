# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:50:52 2017

@author: quentinpeter
"""
import json
import re
import os
from glob import glob
import numpy as np
import warnings
from natsort import natsorted

from .myJSONEncoder import myJSONEncoder, floatstr


class Object(object):
    pass


def _makeabs(prefix, filename):
    """Combine a prefix and a filename to create an absolute path.

    Parameters
    ----------
    prefix: path
        Prefix to combine with the file name. Can be relative.
    filename: path
        Path to the file

    Returns
    -------
    filename: path
        absolute path
    """
    if (filename is not None and
            not os.path.isabs(filename)):
        filename = os.path.join(prefix, filename)
        filename = os.path.abspath(filename)
    return filename


def _listmakeabs(prefix, filename):
    """Combine a prefix and a list of filenames to create absolute paths.

    Parameters
    ----------
    prefix: path
        Prefix to combine with the file name. Can be relative.
    filename: path or list of paths
        Path to the file

    Returns
    -------
    filenames: path or path list
        (list of) absolute path
    """
    if isinstance(filename, (list, tuple)):
        return [_makeabs(prefix, fn) for fn in filename]
    else:
        return _makeabs(prefix, filename)


class ListGenerator():

    def __init__(self, name, example_path=None,
                 data_related=False, data_field=None):
        super().__init__()
        self._list = {}
        self.name = name
        self.data_related = data_related
        self.example_path = example_path
        self.data_field = data_field

    def add_info(self, key, description, dtype,
                 required=True, default=None, example=None,
                 explanation=None, regexp=None, legacy=False):
        info = Object()
        info.key = key
        info.description = description
        info.dtype = dtype
        info.required = required
        info.default = default
        info.example = example
        info.explanation = explanation
        info.regexp = regexp
        info.legacy = legacy
        self._list[description] = info

    def generate_script(self, scriptname):
        with open(scriptname, 'w') as f:
            f.write("from diffusion_device.keys import {}".format(self.name))
            f.write("\n\n\n")
            f.write("# path to data\n")
            f.write("datapath = {}\n".format(repr(self.example_path)))
            f.write("json_infos = {}")
            for key in self._list:
                info = self._list[key]
                if not info.legacy:
                    f.write("\n\n# ")
                    f.write(self._get_comment(info))
                    f.write("\n")
                    example = self._get_repr(self._get_example(info))
                    f.write("json_infos[{}] = {}".format(
                            repr(info.description), example))
            f.write("\n\n")
            f.write("{}.generate_json(datapath, json_infos)".format(self.name))

    def generate_json(self, datapath, json_infos):
        if self.data_related:
            data_list = glob(datapath)
            if data_list == []:
                raise RuntimeError("Can't find {}".format(datapath))
            for datafn in data_list:
                json_infos_copy = json_infos.copy()
                if os.path.isfile(datafn):
                    # Check file name corresponds
                    filename = os.path.basename(datafn)
                    if json_infos_copy[self.data_field] is None:
                        json_infos_copy[self.data_field] = filename
                    elif filename != json_infos_copy[self.data_field]:
                        raise RuntimeError(
                            "Filename mismatch: ['{}'] is not None and '{}' != "
                            "'{}'".format(
                                self.data_field,
                                json_infos_copy[self.data_field],
                                filename))

                self._write_json(datafn, json_infos_copy)
        else:
            self._write_json(datapath, json_infos)

    def load_json(self, filename):
        with open(filename, 'r') as f:
            file = json.load(f)
        ret = {}
        for description in self._list:
            info = self._list[description]
            if info.required:
                self._required(file, info, filename)
            else:
                self._default(file, info)
            self._prepare_read(file, info, filename)
            ret[info.key] = file[info.description]
        return ret

    def _write_json(self, datafn, json_infos):
        metadata = {}
        for description in json_infos:
            info = self._list[description]
            value = json_infos[description]
            value = self._prepare_write(value, info, datafn)
            if info.required or self._value_given(value):
                metadata[description] = value

        filename = self._get_name(datafn)
        with open(filename, 'w') as f:
            json.dump(metadata, f, indent=4, cls=myJSONEncoder)

    def _get_comment(self, info):
        if info.explanation is not None:
            return info.explanation
        return info.description

    def _get_example(self, info):
        if info.example is not None:
            return info.example
        if info.default is not None:
            return info.default
        if not info.required:
            return None
        if info.dtype in [float, int]:
            return 0
        elif info.dtype == str:
            return ""
        return None

    def _get_repr(self, value):
        if isinstance(value, (list, tuple)):
            value = ('[\n    '
                     + ',\n    '.join([self._get_repr(v) for v in value])
                       + ']')
        elif isinstance(value, (float)):
            value = floatstr(value)
        elif isinstance(value, (dict)):
            value = repr(value)
        else:
            value = repr(value)
        return value

    def _cast_type(self, value, dtype):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            value = [self._cast_type(v, dtype) for v in value]
        else:
            value = dtype(value)
        return value

    def _prepare_write(self, value, info, datapath):
        # Regexp
        if value is None and info.regexp is not None:
            allfind = re.findall(info.regexp, datapath)
            if len(allfind) > 0:
                value = allfind[-1]
                if info.dtype == float:
                    value = re.sub('p', '.', value)
        # If still None, skip
        if value is None:
            return None

        if info.dtype in [float, int, str, bool]:
            value = self._cast_type(value, info.dtype)

        elif info.dtype == "path":
            if os.path.isfile(datapath) or not self.data_related:
                pathprefix = os.path.dirname(datapath)
            else:
                pathprefix = datapath

            absdatapath = os.path.join(pathprefix, value)
            value = natsorted(glob(absdatapath))
            if value == []:
                raise RuntimeError("Can't find {}".format(absdatapath))
            value = [os.path.relpath(path, pathprefix) for path in value]
            if len(value) == 0:
                value = None
            elif len(value) == 1:
                value = value[0]
        return value

    def _value_given(self, value):
        if isinstance(value, (list, tuple)):
            return not np.all([v is None for v in value])
        return value is not None

    def _get_name(self, filename):
        if not self.data_related:
            return filename
        if os.path.isfile(filename):
            return os.path.splitext(filename)[0] + '_' + self.name + '.json'
        else:
            return os.path.join(os.path.splitext(
                filename)[0], self.name + '.json')

    def _required(self, file, info, filename):
        if info.description not in file:
            raise RuntimeError("Missing Key: '{}' not in {}".format(
                info.description, filename))

    def _default(self, file, info):
        if info.description not in file or file[info.description] is None:
            file[info.description] = info.default

    def _prepare_read(self, file, info, filename):
        if info.dtype == "path":
            if file[info.description] is not None:
                file[info.description] = _listmakeabs(
                    os.path.dirname(filename), file[info.description])
