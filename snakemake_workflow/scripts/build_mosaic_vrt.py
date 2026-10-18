"""Build a GDAL VRT mosaic of all chunk waterdepth outputs for one SLR scenario."""

from osgeo import gdal

gdal.UseExceptions()

vrt = gdal.BuildVRT(
    str(snakemake.output.waterdepth_vrt),  # noqa: F821
    [str(p) for p in snakemake.input.waterdepth],  # noqa: F821
)
vrt.FlushCache()
del vrt
