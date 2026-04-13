# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:27:48 2017

@author: quentinpeter
"""

from json import encoder
import numpy as np
# Ugly hack to get ENG format


def floatstr(o, _inf=encoder.INFINITY,
             _neginf=-encoder.INFINITY):
    if o != o:
        text = 'NaN'
    elif o == _inf:
        text = 'Infinity'
    elif o == _neginf:
        text = '-Infinity'
    elif o == 0:
        text = '0.0'
    else:
        exp = int(np.floor(np.log10(np.abs(o)) / 3) * 3)
        if exp != 0:
            text = "{:g}e{:d}".format(o / (10**exp), exp)
        else:
            text = "{:g}".format(o)

    return text


class myJSONEncoder(encoder.JSONEncoder):

    def __init__(self, *arg, **kargs):
        super().__init__(*arg, **kargs)

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encoder.encode_basestring_ascii
        else:
            _encoder = encoder.encode_basestring

        def _floatstr(o, allow_nan=self.allow_nan,
                      _repr=float.__repr__, _inf=encoder.INFINITY,
                      _neginf=-encoder.INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            text = floatstr(o, _inf, _neginf)
            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " +
                    repr(o))

            return text

        if (_one_shot and encoder.c_make_encoder is not None
                and self.indent is None):
            _iterencode = encoder.c_make_encoder(
                markers, self.default, _encoder, self.indent,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, self.allow_nan)
        else:
            _iterencode = encoder._make_iterencode(
                markers, self.default, _encoder, self.indent, _floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)
