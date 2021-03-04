"""Module which defines :class:`FbuCCDData`, a fallback unit version of :class:`~astropy.nddata.CCDData`
"""

from astropy import log
from astropy.io import fits
from astropy.nddata import CCDData, fits_ccddata_reader
from astropy.units import Quantity

def fallback_unit_ccddata_reader(filename, *args, 
                                 unit=None,
                                 fallback_unit=None,
                                 **kwargs):
    """Wrapper around `~astropy.nddata.fits_ccddata_reader` to add
    fallback unit capability

    Parameters
    ---------
    filename : str
        Name of FITS file

    unit : `~astropy.units.Unit`, optional
        Units of the image data.   If ``None,`` the FITS header
        ``BUNIT`` keyword will be queried to set the unit.  If that
        is not found or an error is raised, `fallback_unit` will be
        used.
        Default is ``None``.

    fallback_unit : `~astropy.units.Unit`, optional
        Units to be used for the image data if `unit` is not provided
        and no valid ``BUNIT`` value is found in the FITS header.
        Default is ``None``.

    kwargs :
        Keywords to pass to `~astropy.nddata.fits_ccddata_reader`

    """
    if unit is not None:
        return fits_ccddata_reader(filename, *args,
                                   unit=unit, **kwargs)
    # Open the file and read the primary hdr to see if there is a
    # BUNIT there.  Do the open as an memmap to minimize overhead
    # on the second file read in fits_ccddata_reader.  Having the
    # underlying astropy code implement something like
    # hdul_ccddata_reader would save the second file open entirely
    with fits.open(filename, memmap=True) as hdul:
        hdr = hdul[0].header
        bunit = hdr.get('BUNIT')
        if bunit is not None:
            try:
                # fits_ccddata_reader will find BUNIT again.
                # We have to do it without unit=bunit to avoid
                # annoying message
                return fits_ccddata_reader(filename,
                                           *args, unit=unit, **kwargs)
            except ValueError:
                # BUNIT may not be valid
                log.warning(f'Potentially invalid BUNIT '
                            'value {bunit} detected in FITS '
                            'header.  Falling back to {fallback_unit}')
        # If we made it here, there is no BUNIT in the header or it is invalid
        return fits_ccddata_reader(filename, *args,
                                   unit=fallback_unit, **kwargs)


class FbuCCDData(CCDData):
    """
    Add ``fallback_unit`` capability to :meth:`CCDData.read() <~astropy.nddata.CCDData.read>`

    Parameters
    ----------
    filename, unit, fallback_unit : Valid in read method
        See :func:`fallback_unit_ccddata_reader()`

    args, kwargs :
        args and keywords to pass to `~astropy.nddata.CCDData`

    Example
    -------
    >>> class MyCCDData(FbuCCDData):
            def __init__(self, data, my_key=None, **kwargs):
                super().__init__(data, fallback_unit='adu', **kwargs)
                self.my_prop = my_key

    """
    def __init__(self, data,
                 filename=None,
                 unit=None,
                 fallback_unit=None,
                 **kwargs):
        if filename is not None:
            ccd = fallback_unit_ccddata_reader(filename, 
                                               unit=unit,
                                               fallback_unit=fallback_unit,
                                               **kwargs)
            self.__dict__.update(ccd.__dict__)
        else:
        #    # Catch all possible cases where unit is specified in the
        #    # data somehow.  If not and we have not specified our unit
        #    # explicitly, use our fallback unit
        #    try:
        #        # data is already NDData-like
        #        dataunnit = data.unit
        #    except:
        #        dataunnit = None
        #    if (unit is None
        #        and not isinstance(data, Quantity)
        #        and dataunnit is None):
        #        unit = fallback_unit
            super().__init__(data, unit=unit, **kwargs)

    @classmethod
    def read(cls, filename, *args, **kwargs):
        """See `fallback_unit_ccddata_reader`"""
        return cls(None, filename=filename, **kwargs)


if __name__ == "__main__":
    # tests
    log.setLevel('DEBUG')
    rawname = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
    redname = '/data/io/IoIO/reduced/Calibration/2020-07-07_ccdT_-10.3_bias_combined.fits'
    rawccd = FbuCCDData.read(rawname, fallback_unit='adu')
    print(rawccd.unit)
    rawccd = FbuCCDData.read(rawname, unit='electron')
    print(rawccd.unit)
    # Didn't know this feature came along for free!
    rawccd = FbuCCDData(rawccd)
    print(rawccd.unit)
    rawccd = FbuCCDData(rawccd.data, unit='parsec')
    print(rawccd.unit)
    rawccd = FbuCCDData(rawccd.data, fallback_unit='Msun')
    print(rawccd.unit)
    #
    print('========= redccd tests ===========')
    redccd = FbuCCDData.read(redname)
    print(redccd.unit)
    redccd = FbuCCDData.read(redname, unit='adu')
    print(redccd.unit)
    redccd = FbuCCDData.read(redname, unit='electron')
    print(redccd.unit)
    redccd = FbuCCDData.read(redname, fallback_unit='adu')
    print(redccd.unit)
    #
    print('========= unit tests ===========')
    from astropy import units as u
    rawccd = FbuCCDData.read(rawname, fallback_unit='adu')
    print(rawccd.unit)
    gain = 1*u.electron/u.adu
    print(gain)
    rawccd = rawccd.multiply(gain)
    print(rawccd.unit)
    
    ccd = CCDData.read(rawname, unit='adu')
    ccd = ccd.multiply(gain)
    print(ccd.unit)

    print('========= inherited unit tests ===========')


    class RedCorCCDData(FbuCCDData):
        def __init__(self, data, **kwargs):
            super().__init__(data, fallback_unit='adu', **kwargs)
    
    rawccd = RedCorCCDData.read(rawname)
    print(rawccd.unit)
    rawccd = rawccd.multiply(gain)
    print(rawccd.unit)
