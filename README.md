Implements BigMultiPipe subclass for astropy.NDData.CCDData and, in
the utils directory, tweaks the CCDData class to add a fallback_unit
feature, which makes it easier to just read in a FITS file that may or
may not have a BUNIT keyword.
