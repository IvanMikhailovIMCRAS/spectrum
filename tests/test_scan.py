import sys
import os
import numpy as np
import pytest

sys.path.append(os.path.abspath(f'..\\src'))

from src.scan import read_data


SPECTRA_PATHS = [
        'sample_spectra/first_class/SD4.17', 'sample_spectra/first_class/SD6.20',
        'sample_spectra/second_class/SD7.17', 'sample_spectra/second_class/nested_class/SD9.60'
    ]
__OPS = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y
    }

@pytest.mark.parametrize('classify,recursive,result',
    [
        ('not classify', 'non recursive', (1, {'sample_spectra': 1})),
        ('classify', 'non recursive', (3, {'first_class': 2, 'second_class': 1})),
        ('classify', 'non recursive', (3, {'first_class': 2, 'second_class': 1})),
        ('not classify', 'recursive', (5, {'sample_spectra': 5})),
        ('classify', 'recursive', (4, {'first_class': 2, 'second_class': 2})),
    ]
)
def test_read_data(classify, recursive, result):
    res = read_data('sample_spectra', classify=(classify == 'classify'), recursive=(recursive == 'recursive'))
    assert len(res) == result[0], 'The selection went wrong!'
    d = {}
    for i in res:
        d[i[1]] = d.get(i[1], 0) + 1
    assert d == result[1], 'The classification went wrong!'