import operator
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype
from pandas.core.arrays import ExtensionArray


_OCC_DATE_FORMAT = '%y%m%d'


class OccSymbol(object):

    def __init__(self, s):
        self.occ = s


class OccType(ExtensionDtype):
    name = 'occ'
    type = OccSymbol
    kind = 'O'
    _record_type = np.dtype([('symbol', 'U6'), ('expiry', 'M8[D]'), ('otype', '?'), ('strike', 'f4')])

    @classmethod
    def construct_from_string(cls, string):
        if string == cls.name:
            return cls()
        else:
            raise TypeError("Cannot construct a '{}' from "
                            "'{}'".format(cls, string))


def _occ_to_tuple(s):
    symbol, date, put_or_call, strike = s[:6], s[6:12], s[12], s[13:]
    return symbol.rstrip(), pd.Timestamp.strptime(date, _OCC_DATE_FORMAT), put_or_call == 'P', float(strike)/1000


def _to_otype(b):
    return 'P' if b else 'C'


def _pad_symbol(symbol):
    return symbol.ljust(6)


def _strike_to_str(strike):
    return str(int(strike * 1000)).zfill(8)


def _to_occ_str(symbol, expiry, otype, strike):
    return "{}{}{}{}".format(_pad_symbol(symbol), pd.to_datetime(expiry).strftime(_OCC_DATE_FORMAT),
                             _to_otype(otype), _strike_to_str(strike))


def _occ_to_tuples(values):
    return [_occ_to_tuple(v) for v in values]


def _to_occ_array(values):
    values = _occ_to_tuples(values)
    return np.atleast_1d(np.asarray(values, dtype=OccType._record_type))


class OccArray(ExtensionArray):
    dtype = OccType
    _itemsize = OccType._record_type.itemsize

    def __init__(self, values):
        values = _to_occ_array(values)
        self.data = values

    @property
    def nbytes(self):
        return self._itemsize * len(self)

    @classmethod
    def _from_sequence(cls, scalars):
        return cls(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values)

    def __getitem__(self, *args):
        result = operator.getitem(self.data, *args)
        if isinstance(result, tuple):
            return _to_occ_str(*result)
        elif result.ndim == 0:
            return _to_occ_str(*result.item())
        else:
            return type(self)(result)

    def __len__(self):
        return len(self.data)

    def isna(self):
        occs = self.data
        return (occs['symbol'] == 0) & (occs['expiry'] == '1970-01-01') & (occs['otype'] == 0) & (occs['strike'] == 0)

    def take(self, indices, allow_fill=False, fill_value=None):
        pass

    def copy(self, deep=False):
        return type(self)(self.data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        return cls(np.concatenate([array.data for array in to_concat]))

    def __repr__(self):
        formatted = self._format_values()
        return "OccArray({!r})".format(formatted)

    def _format_values(self):
        formatted = []
        for i in range(len(self)):
            symbol, expiry, put_or_call, strike = self.data[i]
            formatted.append(_to_occ_str(symbol, expiry, put_or_call, strike))
        return formatted

    @property
    def is_call(self):
        return self.data['otype'] == 0

    @property
    def is_put(self):
        return self.data['otype'] == 1

    def is_expired(self, date=None):
        if not date:
            date = pd.Timestamp.today()
        return pd.to_datetime(self.data['expiry']) < date


def delegated_method(method, index, name, *args, **kwargs):
    return pd.Series(method(*args, **kwargs), index, name=name)


class Delegated:
    # Descriptor for delegating attribute access to from
    # a Series to an underlying array

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        result = self._get_result(obj)
        return pd.Series(result, index, name=name)


class DelegatedProperty(Delegated):
    def _get_result(self, obj, type=None):
        return getattr(object.__getattribute__(obj, '_data'), self.name)


class DelegatedMethod(Delegated):
    def __get__(self, obj, type=None):
        index = object.__getattribute__(obj, '_index')
        name = object.__getattribute__(obj, '_name')
        method = getattr(object.__getattribute__(obj, '_data'), self.name)
        return delegated_method(method, index, name)


@pd.api.extensions.register_series_accessor('occ')
class OccAccessor:

    is_call = DelegatedProperty("is_call")
    is_put = DelegatedProperty("is_put")

    def __init__(self, obj):
        self._validate(obj)
        self._data = obj.values
        self._index = obj.index
        self._name = obj.name

    @staticmethod
    def _validate(obj):
        return isinstance(obj.dtype, OccType)

    def is_expired(self, date=None):
        return delegated_method(self._data.is_expired, self._index, self._name, date)
