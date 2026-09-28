"""Minimal stand-in for `tensorflow` so ScenarioNet's Waymo converter can read .tfrecord files without installing TF
(only tf.data.TFRecordDataset(path, compression_type="").as_numpy_iterator() is used).  Put src/shim on PYTHONPATH."""
import struct


class _TFRecordDataset:
    def __init__(self, path, compression_type=""):
        assert not compression_type, "compressed tfrecords not supported by shim"
        self.path = path

    def as_numpy_iterator(self):
        with open(self.path, "rb") as f:
            while True:
                head = f.read(8)
                if len(head) < 8:
                    return
                (n,) = struct.unpack("<Q", head)
                f.read(4)                       # masked crc32c of length (not verified)
                data = f.read(n)
                if len(data) < n:
                    raise IOError("truncated tfrecord: " + self.path)
                f.read(4)                       # masked crc32c of data
                yield data


class data:
    TFRecordDataset = _TFRecordDataset


class config:
    class experimental:
        @staticmethod
        def set_visible_devices(*a, **k):
            return None
