"""Module which enables parallel pipeline processing of CCDData files

"""

import os
import argparse
import warnings

import numpy as np

from astropy import log
from astropy import units as u
from astropy.nddata import CCDData
from astropy.wcs import FITSFixedWarning

from bigmultipipe import BigMultiPipe

OUTNAME_APPEND = '_ccdmp'

# This subclass of CCDData is a better choice than ccddata_read
# from utils.fallback_unit import FbuCCDData

#def ccddata_read(fname_or_ccd,
#                 raw_unit=u.adu,
#                 *args, **kwargs):
#    """Convenience function to read a FITS file into a CCDData object.
#
#    Catches the case where the raw FITS file does not have a BUNIT
#    keyword, which otherwise causes :meth:`CCDData.read` to crash.  In
#    this case, :func:`ccddata_read` assigns `ccd` units of
#    ``raw_unit``.  Also ads comment to BUNIT "physical units of the
#    array values," which is curiously omitted in the astropy fits
#    writing system.  Comment is from `official FITS documentation
#    <https://heasarc.gsfc.nasa.gov/docs/fcg/standard_dict.html>` where
#    BUNIT is in the same family as BZERO and BSCALE
#
#    Parameters
#    ----------
#    fname_or_ccd : str or `~astropy.nddata.CCDData`
#        If str, assumed to be a filename, which is read into a
#        `~astropy.nddata.CCDData`.  If `~astropy.nddata.CCDData`,
#        return a copy of the CCDData with BUNIT keyword possibly 
#        added
#
#    raw_unit : str or `astropy.units.core.UnitBase`
#        Physical unit of pixel in case none is specified 
#        Default is `astropy.units.adu`
#
#    *args and **kwargs passed to :meth:`CCDData.read
#         <astropy.nddata.CCDData.read>` 
#
#    Returns
#    -------
#    ccd : `~astropy.nddata.CCDData`
#        `~astropy.nddata.CCDData` with units set to raw_unit if none
#        specified in FITS file 
#
#    """
#    if isinstance(fname_or_ccd, str):
#        try:
#            ccd = CCDData.read(fname_or_ccd, *args, **kwargs)
#        except Exception as e: 
#            ccd = CCDData.read(fname_or_ccd, *args,
#                               unit=raw_unit, **kwargs)
#    else:
#        ccd = fname_or_ccd.copy()
#    assert isinstance(ccd, CCDData)
#    if ccd.unit is None:
#        log.warning('ccddata_read: CCDData.read read a file and did not assign any units to it.  Not sure why.  Setting units to' + raw_unit.to_string())
#        ccd.unit = raw_unit
#    # Setting ccd.unit to something does not set the BUNIT keyword
#    # until file write.  So to write the comment before we write the
#    # file, we need to set BUNIT ourselves.  If ccd,unit changes
#    # during our calculations (e.g. gain correction), the BUNIT
#    # keyword is changed but the comment is not.
#    ccd.meta['BUNIT'] = (ccd.unit.to_string(),
#                         'physical units of the array values')
#    return ccd

class CCDMultiPipe(BigMultiPipe):
    """Base class for `ccdproc` multiprocessing pipelines

    Parameters
    ----------
    ccddata_cls : `~astropy.nddata.CCDData`-like or ``None``
        `~astropy.nddata.CCDData` or subclass thereof to contain the
        data.  If ``None`` `~astropy.nddata.CCDData` is used
        Default is None

    process_size : int
        Maximum process size in bytes of an individual process.  If
        ``None``, calculated from ``naxisN``, ``bitpix`` and
        ``process_expand_factor`` parameters.
        Default is ``None``

    naxis1, naxis2 : int or None
        CCD image size.  If ``None``, read from *first* image
        processed in pipeline.
        Default is ``None``

    bitpix : int
        Maximum bits per pixel during processing.
        Default is 2*64 + 8 = 136, which includes primary and error
        image as double-precision and mask image as byte

    process_expand_factor : float
        Factor to account for the fact that
        :func:`ccdproc.ccd_process` and each of the routines it calls
        make a copy of the data passed to them.  Thus, at any one time
        their can be up to 3 copies of the :param:`ccddata_cls`
        Default is 3.5

    outname_append : str, optional
        String to append to outname to avoid risk of input file
        overwrite.  Example input file ``test.fits`` would become
        output file ``test_ccdmp.fits``
        Default is ``_ccdmp``

    overwrite : bool
        Set to ``True`` to enable overwriting of output files of the
        same name.
        Default is ``False``

    fits_fixed_ignore : bool

        Set to ``True`` to avoid excessive warnings when processing
        files with metadata that don't conform precisely to current
        astropy/WCS FITS standard keywords (e.g. RADECSYS is now
        RADESYS).  Because :meth:`CCDMultipipe.pipeline()` starts a
        separate process for each each file, the default astropy
        mechanism for limiting warnings:
        warnings.filterwarnings('once', FITSFixedWarning) only works
        once per process, which, unless files are grouped in the input
        in_names, means that every file will generate a warning, thus
        nullifying the default mechanism for quieting the warnings.
        Default is ``False``


    warning_ignore_list : list

        List of warning objects whose warnings will be ignored.  For
        convenience, the ``fits_fits_ignore`` parameter can be used to
        put :class:`astropy.wcsFITSFixedWarning` on this list.  See
        ``fits_fixed_ignore`` for detailed discussion.  Default is
        ``[]``

    kwargs : kwargs
        Passed to __init__ method of :class:`bigmultipipe.BigMultiPipe`

    """

    ccddata_cls = CCDData
    def __init__(self,
                 ccddata_cls=None,
                 process_size=None,
                 naxis1=None,
                 naxis2=None,
                 bitpix=None,
                 process_expand_factor=3.5,
                 outname_append=OUTNAME_APPEND,
                 overwrite=False,
                 fits_fixed_ignore=False,
                 warning_ignore_list=[],
                 **kwargs):
        self.ccddata_cls = ccddata_cls or self.ccddata_cls
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        if bitpix is None:
            bitpix = 2*64 + 8
        self.bitpix = bitpix
        self.process_expand_factor = process_expand_factor
        self.process_size = process_size
        self.overwrite = overwrite
        self.fits_fixed_ignore = fits_fixed_ignore
        self.warning_ignore_list = warning_ignore_list
        super().__init__(outname_append=outname_append,
                         **kwargs)

    def pipeline(self, in_names,
                 process_size=None,
                 naxis1=None,
                 naxis2=None,
                 bitpix=None,
                 process_expand_factor=None,
                 fits_fixed_ignore=None,
                 warning_ignore_list=None,
                 **kwargs):
        """Runs pipeline, maximizing processing and memory resources

        Parameters
        ----------
        in_names : `list` of `str`
            List of input filenames.  Each file is processed using
            :func:`file_process`

        All other parameters : see Parameters to :class:`CCDMultiPipe`
        and :class:`bigmultipipe.BigMultiPipe` 

        Returns
        -------
        pout : `list` of tuples ``(outname, meta)``, one `tuple` for each
            ``in_name``.  ``Outname`` is `str` or ``None``.  If `str`,
            it is the name of the file to which the processed data
            were written.  If ``None``, the convenience function
            :func:`prune_pout` can be used to remove this tuple from
            ``pout`` and the corresponding in_name from the in_names list.
            ``Meta`` is a `dict` containing output.

        """
        if process_size is None:
            process_size = self.process_size
        if naxis1 is None:
            naxis1 = self.naxis1
        if naxis2 is None:
            naxis2 = self.naxis2
        if bitpix is None:
            bitpix = self.bitpix
        if process_expand_factor is None:
            process_expand_factor = self.process_expand_factor
        if (process_size is None):
            if naxis1 is None or naxis2 is None:
                ccd = self.file_read(in_names[0], **kwargs)
            if naxis1 is None:
                naxis1 = ccd.meta['NAXIS1']
            if naxis2 is None:
                naxis2 = ccd.meta['NAXIS2']
            process_size = (naxis1 * naxis2
                            * bitpix/8
                            * process_expand_factor)
        if fits_fixed_ignore is None:
            fits_fixed_ignore = self.fits_fixed_ignore
        if warning_ignore_list is None:
            warning_ignore_list = self.warning_ignore_list


        if fits_fixed_ignore:
            warning_ignore_list.append(FITSFixedWarning)
            
        with warnings.catch_warnings():
            for w in warning_ignore_list:
                warnings.filterwarnings("ignore", category=w)
            return super().pipeline(in_names, process_size=process_size,
                                    **kwargs)

    def file_read(self, in_name, **kwargs):
        """Reads FITS file(s) from disk

        Parameters
        ----------
        in_name : str or list
            If `str`, name of FITS file to read.  If `list`, each
            element in list is processed recursively so that multiple
            files can be considered a single "data" in `bigmultipipe`
            nomenclature

        kwargs : kwargs
            kwargs for any underlying routines.  However, *all*
            routines must accept all kwargs, so if underlying routines
            can only handle specific kwargs, they must be accepted to
            file_read and passed explicitly to the underlying routines

        Returns
        -------
        data : :class:`~astropy.nddata.CCDData`
            :class:`~astropy.nddata.CCDData` to be processed

        """
        kwargs = self.kwargs_merge(**kwargs)
        if isinstance(in_name, str):
            return self.ccddata_cls.read(in_name)
        # Allow list (of lists...) to be read into a "data"
        return [self.file_read(name, **kwargs)
                for name in in_name]

    # Have to implement loop in loops in primitive
    #def pre_process(self, data, **kwargs):
    #    """Enable list of CCDData to be processed one at a time"""
    #    if isinstance(data, list):
    #        return[self.pre_process(d, **kwargs)
    #               for d in data]
    #    return super().pre_process(data, **kwargs)
    #
    #def post_process(self, data, kwargs):
    #    if isinstance(data, list):
    #        return[self.post_process(d, **kwargs)
    #               for d in data]
    #    return super().post_process(data, **kwargs)

    def outname_create(self, *args,
                       outname_ext=None,
                       **kwargs):
        """Create output filename (including path)

        Parameters
        ----------
        \*args : required
            Passed to :meth:`bigmultipipe.Bigmultipipe.outname_create`
            (e.g. in_name)

        outname_ext : str or None, optional
            Extension of output filename.  Useful for accepting .fts
            files as input and writing them as .fits files
            Default is ``None``

        \*\*kwargs : optional
            Passed to :meth:`bigmultipipe.Bigmultipipe.outname_create` 
        """
        outname = super().outname_create(*args, **kwargs)
        if outname_ext is None:
            return outname
        base, _ = os.path.splitext(outname)
        outname = base + outname_ext
        return outname

    def file_write(self, data, outname,
                   as_scaled_int=False,
                   bscale=None,
                   bzero=None,
                   overwrite=None,
                   ccddata_write=True,
                   **kwargs):
        """Write `~astropy.nddata.CCDData` as FITS file file.

        Parameters
        ----------
        data : `~astropy.nddata.CCDData`
            Processed data

        outname : str
            Name of file to write

        as_scaled_int : bool
            If ``True`` convert floating point extensions to 16-bit
            scaled integers as per ``bscale`` or ``bzero``.  If
            ``bscale`` and ``bzero`` are not specified, they are
            created using min and max.  See Notes and as_single
            post-processing routine

        bscale, bzero : number or None
            When scaling with as_scaled_int, these are used to scale
            data from physical values to int16: 
            physical value = BSCALE * (storage value) + BZERO
            If ``None``, created using data min and max.
            Default is ``None``

        overwrite : bool
            If ``True`` overwrite existing file of same name
            Default is ``False``

        ccddata_write : bool
            If ``False`` do not write `~astropy.nddata.CCDData` file.
            A similar effect can be achieved with the
            `bigmultipipe.no_outfile` post-processing routine.  This simply
            enables the feature with a kwarg.
            Default is ``True``

        kwargs : kwargs

        Returns
        -------
        outname : str
            Name of file written, `None` if no file written

        Notes
        -----

        Use of ``as_scaled_int`` is officially discouraged by the
        `astropy.io.fits.PrimaryHDU` documentation because of expense
        in computation and memory.  However, when viewed as a means of
        compressing data, this argument falls short: all compression
        takes some amount of computation and memory.  For typical
        image data, gzip provides compression at the 90% level;
        ``as_scaled_int`` provides compression of a factor of 4
        (assuming doubles as input).  However, significant loss of
        precision will occur with ``as_scaled_int`` if care is not
        taken to clip high and low values.  

        The post-processing routine as_single provides a compression of a
        factor of 2, reasonable data precision, and no need to clip.

        """
        kwargs = self.kwargs_merge(**kwargs)
        if not ccddata_write:
            return None
        overwrite = overwrite or self.overwrite
        if as_scaled_int:
            hdul = data.to_hdu()
            hdul[0].scale(type='int16', option='minmax')
            if data.uncertainty is not None:
                iuncert = hdul.index_of('UNCERT')
                hdul[iuncert].scale(type='int16', option='minmax',
                                    bscale=bscale, bzero=bzero)
            hdul.writeto(outname, overwrite=overwrite)
        else:
            data.write(outname, overwrite=overwrite)
        return outname

#############################
# Post-processing routines
#############################
def as_single(ccd_in,
              as_single_datatype=None,
              **kwargs):
    """CCDMultiPipe post-processing routine to convert ccd data and
    uncertainty arrays to an alternate (usually smaller)
    datatype

    Parameters
    ----------
    ccd_in : `~astropy.nddata.CCDData`
        Input `~astropy.nddata.CCDData` object

    as_single_datatype : `numpy` datatype
        Datatype to convert primary and uncertainty arrays to.  If
       ``None``, converts to `numpy.single`
        Default is ``None``

    """
    if as_single_datatype is None:
        as_single_datatype = np.single
    ccd = ccd_in.copy()
    ccd.data = ccd.data.astype(as_single_datatype)
    ccd.uncertainty.array = ccd.uncertainty.array.astype(as_single_datatype)
    return ccd

    # Example data_process using ccdproc.ccd_process, e.g.
    #import ccdproc as ccdp
    #def data_process(self, data, **kwargs):
    #    """Process data using :func:`ccdproc.ccd_process`
    #    
    #    Parameters
    #    ----------
    #    data : `~astropy.nddata.CCDData`
    #        Data to process
    #
    #    kwargs : kwargs
    #        Passed to :func:`ccdproc.ccd_process`
    #        See also: Notes in :class:`bigmultipipe.BigMultiPipe`
    #        Parameter section 
    #
    #    """
    #    kwargs = self.kwargs_merge(**kwargs)
    #    data = ccdp.ccd_process(data, **kwargs)
    #    return data

#############################
# argparse_handler mixin
#############################
class CCDArgparseMixin:

    # This modifies BMPArgparseMixin outname_append
    outname_append = OUTNAME_APPEND

    def add_naxis1(self, 
                   default=None,
                   help=None,
                   **kwargs):
        option = 'naxis1'
        if help is None:
            help = 'number of X-axis pixels'
        self.parser.add_argument('--' + option, type=int,
                            default=default, help=help, **kwargs)

    def add_naxis2(self, 
                   default=None,
                   help=None,
                   **kwargs):
        option = 'naxis2'
        if help is None:
            help = 'number of Y-axis pixels'
        self.parser.add_argument('--' + option, type=int,
                            default=default, help=help, **kwargs)

    def add_bitpix(self, 
                   help=None,
                   default=8,
                   **kwargs):
        option = 'bitpix'
        if help is None:
            help = f'number of bits per pixel (default: {default})'
        self.parser.add_argument('--' + option, type=int,
                            default=default, help=help, **kwargs)

    def add_process_expand_factor(self, 
                                  default=None,
                                  help=None,
                                  **kwargs):
        option = 'process_expand_factor'
        if help is None:
            help = 'Expansion factor to apply to base CCD size'
        self.parser.add_argument('--' + option, type=float,
                            default=default, help=help, **kwargs)

    def add_overwrite(self, 
                      default=False,
                      help=None,
                      **kwargs):
        option = 'overwrite'
        if help is None:
            help = (f'Overwrite output files of the same name')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

    def add_fits_fixed_ignore(self, 
                      default=False,
                      help=None,
                      **kwargs):
        option = 'fits_fixed_ignore'
        if help is None:
            help = (f'Ignore fits_fixed warnings')
        self.parser.add_argument('--' + option,
                                 action=argparse.BooleanOptionalAction,
                                 default=default,
                                 help=help, **kwargs)

#bname = '/data/io/IoIO/reduced/Calibration/2020-07-07_ccdT_-10.3_bias_combined.fits'
#ccd = CCDData.read(bname)

#fname1 = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
#ccd = CCDData.read(fname1)
#ccd = FBUCCDData.read(fname1, fallback_unit='adu')
#ccd = FBUCCDData.read(fname1, fallback_unit='aduu')

#ccd = FBUCCDData.read(fname1, unit='electron')
