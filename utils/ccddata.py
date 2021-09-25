"""Supplements to astropy.nddata.ccddata
"""

import warnings

from astropy import log
from astropy.io import fits
from astropy.nddata import CCDData, fits_ccddata_reader

class FilterWarningCCDData(CCDData):
    warning_filter_list = []

    def __init__(self, *args,
                 warning_filter_list=None,
                 **kwargs):
        self.warning_filter_list = (warning_filter_list
                                    or self.warning_filter_list)
        with warnings.catch_warnings():
            for w in self.warning_filter_list:
                warnings.filterwarnings("ignore", category=w)
            super().__init__(*args, **kwargs)

    @classmethod
    def read(cls, *args, **kwargs):
        with warnings.catch_warnings():
            for w in cls.warning_filter_list:
                warnings.filterwarnings("ignore", category=w)
            return super(FilterWarningCCDData, cls).read(*args, **kwargs)
    


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
        ``BUNIT`` keyword in metadata will be queried to set the unit.
	If that is not found or an error is raised querying it,
        `fallback_unit`  will be used.
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
    """Enable 3-level treatment of ``unit`` specification for
    `~astropy.nddata.CCDData`:
    (1) Force: specify non-``None`` ``unit``
    (2) Inspect: Use BUNIT value from `~astropy.nddata.CCDData` metadata.
    (3) Fallback/default: specify non-``None`` ``fallback_unit``

    Parameters
    ----------
    unit : `~astropy.units.Unit`, optional
        Units of the image data.   If ``None,`` the FITS header
        ``BUNIT`` keyword in metadata will be queried to set the unit.
	If that is not found or an error is raised querying it,
        `fallback_unit`  will be used.
        Default is ``None``.

    fallback_unit : `~astropy.units.Unit`, optional
        Units to be used for the image data if `unit` is not provided
        and no valid ``BUNIT`` value is found in the FITS header.
        Default is ``None``.

    args, kwargs :
        args and keywords to pass to `~astropy.nddata.CCDData`

    Example
    -------
    Here are two examples of how to make a
    `~astropy.nddata.CCDData`-like class always default to ``adu`` if
    ``unit`` is not specified in any other way.

    >>> class MyCCDData(FbuCCDData:
    >>>     fallback_unit = 'adu'
    >>>     ....

    or

    >>> class ADU():
    >>>     fallback_unit = 'adu'
    >>> 
    >>> class MyCCDData(ADU, FbuCCDData):
    >>>     ....

    """
    fallback_unit = None
    ccddata_reader = fallback_unit_ccddata_reader
    def __init__(self, data,
                 filename=None,
                 unit=None,
                 fallback_unit=None,
                 ccddata_reader=None,
                 meta=None,
                 **kwargs):
        fallback_unit = fallback_unit or self.fallback_unit
        ccddata_reader = ccddata_reader or self.ccddata_reader
        # Enable fallback_unit in direct instantiation case.  This
        # requires us to catch the case where the data is already a
        # Quantity or NDData-like and meta may has a BUNIT keyword
        if hasattr(data, 'unit'):
            dataunnit = data.unit
        else:
            dataunnit = None
        if meta:
            hdrunit = meta.get('BUNIT')
        else:
            hdrunit = None
        unit = unit or dataunnit or hdrunit or fallback_unit
        super().__init__(data, unit=unit, meta=meta, **kwargs)

    @classmethod
    def read(cls, filename, 
             fallback_unit=None,
             **kwargs):
        """See `fallback_unit_ccddata_reader`"""
        fallback_unit = fallback_unit or cls.fallback_unit
        ccd = cls.ccddata_reader(filename,
                                 fallback_unit=fallback_unit,
                                 **kwargs)
        return cls(ccd, **kwargs)

