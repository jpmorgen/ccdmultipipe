"""Module which adds fallback_unit to :meth:`CCDData.read() <~astropy.nddata.CCDData.read>`
"""

from astropy import log
from astropy.io import fits
from astropy.nddata import CCDData, fits_ccddata_reader

def fallback_unit_ccddata_reader(filename, *args, 
                                 unit=None,
                                 fallback_unit=None,
                                 **kwargs):
    """Wrapper around `~astropy.nddata.fits_ccddata_reader` to add fallback unit capability

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
                return fits_ccddata_reader(filename, *args, **kwargs)
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
            def __init__(self, *args, **kwargs):
                super().__init__(*args, fallback_unit='adu', **kwargs)

    """
    def __init__(self, *args,
                 filename=None,
                 fallback_unit=None,
                 **kwargs):
        if filename is not None:
            ccd = fallback_unit_ccddata_reader(filename, *args, 
                                               fallback_unit=fallback_unit,
                                               **kwargs)
            self.__dict__.update(ccd.__dict__)
        else:
            super().__init__(*args, **kwargs)

    @classmethod
    def read(cls, filename, *args, **kwargs):
        """See `fallback_unit_ccddata_reader`"""
        return cls(*args, filename=filename, **kwargs)
