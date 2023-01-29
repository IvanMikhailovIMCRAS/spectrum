import os
import sys
import numpy as np
import pytest
from enums import Scale
import exceptions
import spectrum
from random import randint
SPECTRA_PATHS = [
        'sample_spectra/first_class/SD4.17', 'sample_spectra/first_class/SD6.20',
        'sample_spectra/second_class/SD7.17', 'sample_spectra/second_class/nested_class/SD9.60'
    ]
__OPS = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y
    }



def test_filter_opus():
    assert spectrum.filter_opus('sample_spectra/SD1.26'), 'Opus files should be available!'
    assert not spectrum.filter_opus('sample_spectra/garbage.txt'),\
        'Files with not digit-like extention must be filtered off!'
    assert not spectrum.filter_opus('sample_spectra/broken_file.21'),\
        'Only binary undamaged files are allowed!'


@pytest.fixture(scope='module')
def sample_spectra():
    res = []
    for path in SPECTRA_PATHS:
        res.append(spectrum.Spectrum(path=path, clss='undefined'))
    return res


def test_operations(sample_spectra):
    sp1 = sample_spectra[randint(0, len(SPECTRA_PATHS) - 1)]
    sp2 = sample_spectra[randint(0, len(SPECTRA_PATHS) - 1)]
    for operation in __OPS:
        res = __OPS[operation](sp1, sp2)
        data = np.array([__OPS[operation](sp1.data[i], sp2.data[i]) for i in range(len(sp1))])
        assert all(data == res.data), f'Operation {operation} result isn\'t right!'
        assert res.id != sp1.id, 'Not in-place operations should lead to creation of a spectrum with its own id!'
        assert sp1._path == res._path, 'Path should be inherited from the first spectrum!'


def test_const_operations(sample_spectra):
    sp1 = sample_spectra[randint(0, len(SPECTRA_PATHS) - 1)]
    const1 = 4.
    for operation in __OPS:
        sp2 = __OPS[operation](sp1, const1)
        sp3 = __OPS[operation](const1, sp1)
        data2 = np.array([__OPS[operation](sp1.data, const1)])
        data3 = np.array([__OPS[operation](const1, sp1.data)])
        assert (data2 == sp2.data).all(), f'Operation {operation} for Spectrum - number pair is wrong!'
        assert (data3 == sp3.data).all(), f'Operation {operation} for reversed operand order (number  - Spectrum) pair is wrong!'
        assert sp3.id != sp2.id and sp2.id != sp1.id, 'Not in-place operations should lead to creation of a spectrum with its own id!'
        assert sp3._path == sp2._path == sp1._path, 'Path should be inherited from the only spectrum used!'


def test_inplace_operations(sample_spectra):
    sp1 = sample_spectra[randint(0, len(SPECTRA_PATHS) - 1)]
    id_sp = sp1.id
    path_sp = sp1._path

    sp2 = sample_spectra[1]
    sp1 += sp2
    assert sp1.id == id_sp and sp1._path == path_sp, 'First spectrum characteristics shouldn\'t be altered!'


@pytest.fixture()
def simplified_spectrum():
    return spectrum.Spectrum(wavenums=np.array(list(range(1, 11)), dtype=float),
                             data=np.array([
                                 -10., -7., 4., -8., -6., 2., 3., 2., 7., 6.
                             ]))


@pytest.mark.parametrize('locals,minima,include_edges,result',
                         [
                             (True, True, True, [0, 3, 7, 9]),
                             (True, True, False, [3, 7]),
                             (True, False, True, [2, 6, 8]),
                             (True, False, False, [2, 6, 8]),
                             (False, True, True, [0]),
                             (False, True, False, [3]),
                             (False, False, True, [8]),
                             (False, False, False, [8]),
                         ]
)
def test_get_extrema(simplified_spectrum, locals, minima, include_edges, result):
    assert simplified_spectrum.get_extrema(locals=locals, minima=minima, include_edges=include_edges)[0] == result, \
        f'{"Local" if locals else "Global" } {"minima" if minima else "maxima"}' \
        f' were found incorrectly! ({"Including" if include_edges else "Excluding"} edges)'


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
    res = spectrum.read_data('sample_spectra', classify=(classify == 'classify'), recursive=(recursive == 'recursive'))
    assert len(res) == result[0], 'The selection went wrong!'
    d = {}
    for i in res:
        d[i[1]] = d.get(i[1], 0) + 1
    assert d == result[1], 'The classification went wrong!'


def test_init(sample_spectra):
    spc = sample_spectra[randint(0, len(sample_spectra) - 1)]
    path, wavenums, data, clss = spc._path, spc.wavenums[:], spc.data[:], spc.clss
    test_spc = spc * 1.
    assert all(test_spc.data == spc.data)\
           and all(test_spc.wavenums == spc.wavenums) \
           and test_spc.clss == 'undefined', \
            'Reading spectrum by a path to an opus file should lead to setting of wavenums and intensities.'
    test_spc = spectrum.Spectrum(data=data, wavenums=wavenums)
    assert all(test_spc.data == spc.data) and all(test_spc.wavenums == spc.wavenums), \
        'Definition by wavenums and intensities should lead to setting of passed ones.'

    with pytest.raises(exceptions.SpcCreationEx):
        spectrum.Spectrum(data=data, wavenums=np.array([], dtype=float))
    with pytest.raises(exceptions.SpcCreationEx):
        spectrum.Spectrum(data=data, wavenums=wavenums[2:])
    with pytest.raises(exceptions.SpcCreationEx):
        spectrum.Spectrum(clss='I\'ve got no roots!',)

@pytest.mark.parametrize('scale_type', [ Scale.WAVELENGTH_nm, Scale.WAVELENGTH_um, Scale.WAVENUMBERS,])
def test_save_as_csv(simplified_spectrum, scale_type):
    tmp_path = 'tmp.csv'
    simplified_spectrum.save_as_csv(tmp_path, scale_type)
    with open(tmp_path, 'r') as file:
        f_line = file.readline().strip()
        s_line = file.readline().strip()
        f = spectrum.scale_change(scale_type)
        scale = [str(f(i)) for i in simplified_spectrum.wavenums]
        expected_first_line = ','.join([scale_type.value] + scale)
        expected_second_line = ','.join([simplified_spectrum.clss] + [str(i) for i in simplified_spectrum.data])
        assert f_line == expected_first_line
        assert s_line == expected_second_line, \
            'The saving should write the proper values!'
    os.remove(tmp_path)

def test_read_csv(simplified_spectrum):
    tmp_path = 'tmp.csv'
    simplified_spectrum.save_as_csv(tmp_path)
    wavenums, data, clss = spectrum.Spectrum.read_csv(tmp_path)
    assert all(wavenums == simplified_spectrum.wavenums) and all(data == simplified_spectrum.data)\
           and clss == simplified_spectrum.clss,\
        'The read file should match the sample one!'
    os.remove(tmp_path)


def test_reset():
    sp_opus = spectrum.Spectrum(path='tests/sample_spectra/SD1.26')
    sp_csv = spectrum.Spectrum(path='sample_spectra/diabetes_1.csv')
    sp_w, sp_d = sp_opus.wavenums.copy(), sp_opus.data.copy()
    csv_w, csv_d = sp_csv.wavenums.copy(), sp_csv.data.copy()

    sp_csv += sp_csv * 3.
    sp_csv.wavenums = sp_csv.wavenums[:23]
    sp_opus.data = sp_csv.data
    sp_opus.wavenums = sp_opus - 0.2

    sp_csv.reset()
    sp_opus.reset()

    assert (sp_w == sp_opus.wavenums) and (sp_d == sp_opus.data), 'The opus reset went wrong!'
    assert (csv_w == sp_csv.wavenums).all() and (csv_d == sp_csv.data).all(), 'The csv reset went wrong!'

#def range(self, bigger, lesser)
