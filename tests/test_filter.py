import pytest
from src.filter import filter_opus

def test_filter_opus():
    assert filter_opus('sample_spectra/SD1.26'), 'Opus files should be available!'
    assert not filter_opus('sample_spectra/garbage.txt'),\
        'Files with not digit-like extention must be filtered off!'
    assert not filter_opus('sample_spectra/broken_file.21'),\
        'Only binary undamaged files are allowed!'