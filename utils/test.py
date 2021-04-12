from astropy import log

from fallback_unit import FbuCCDData

# tests
log.setLevel('DEBUG')
rawname = '/data/Mercury/raw/2020-05-27/Mercury-0005_Na-on.fit'
redname = '/data/io/IoIO/reduced/Calibration/2020-07-07_ccdT_-10.3_bias_combined.fits'
rawccd = FbuCCDData.read(rawname, fallback_unit='adu')
print(f'does {rawccd.unit} = adu?')
rawccd = FbuCCDData.read(rawname, unit='electron')
print(f'does {rawccd.unit} = electron?')
# Didn't know this feature came along for free!
rawccd = FbuCCDData(rawccd)
print(f'does {rawccd.unit} = electron?')
rawccd = FbuCCDData(rawccd.data, unit='parsec')
print(f'does {rawccd.unit} = pc?')
rawccd = FbuCCDData(rawccd.data, fallback_unit='Msun')
print(f'does {rawccd.unit} = Msun?')
#
print('========= redccd tests ===========')
redccd = FbuCCDData.read(redname)
print(f'does {redccd.unit} = electron?')
redccd = FbuCCDData.read(redname, unit='adu')
print(f'does {redccd.unit} = adu?')
redccd = FbuCCDData.read(redname, unit='electron')
print(f'does {redccd.unit} = electron?')
redccd = FbuCCDData.read(redname, fallback_unit='adu')
print(f'does {redccd.unit} = electron?')
#
print('========= unit tests ===========')
from astropy import units as u
rawccd = FbuCCDData.read(rawname, fallback_unit='adu')
print(f'does {rawccd.unit} = adu?')
gain = 1*u.electron/u.adu
rawccd = rawccd.multiply(gain)
print(f'does {rawccd.unit} = electron?')

print('========= inherited unit tests ===========')
class AduCCDData(FbuCCDData):
    fallback_unit = 'adu'
        
rawccd = AduCCDData.read(rawname)
print(f'does {rawccd.unit} = adu?')
rawccd = rawccd.multiply(gain)
print(f'does {rawccd.unit} = electron?')

print('========= hdr tests ===========')
rawccd = FbuCCDData.read(rawname, fallback_unit='adu')
tccdname = '/tmp/rawccd.fits'
rawccd.write(tccdname, overwrite=True)
rawccd = FbuCCDData.read(tccdname, fallback_unit='adu')
hdrccd = FbuCCDData(rawccd.data, meta=rawccd.meta)
print(f'does {hdrccd.unit} = adu?')
