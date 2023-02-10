import logging
import os

class SpectrumEx(Exception):
    logging.basicConfig(level=logging.INFO,
                        filemode='w',
                        filename=os.path.join(os.getcwd(), 'errlog.txt'),
                        format='%(asctime)s %(levelname)s %(message)s')
    def __init__(self, *args):
        super().__init__(*args)
        msg = self.message if self.message else 'Error!'
        logging.error(msg, exc_info=True)

    def __str__(self):
        return f'Error occured! {self.message}'

class SpcChangeEx(SpectrumEx):
    def __init__(self, *args):
        self.message = args[0] if args else 'Spectra should have the same wavenumber ranges!'

class SpcReadingEx(SpectrumEx):
    def __init__(self, *args):
        self.message = args[0] if args else 'File has been damaged. Failed to read.'

class SpcCreationEx(SpectrumEx):
    def __init__(self, *args):
        self.message = args[0] if args else 'Invalid set of parameters!'



