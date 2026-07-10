"""Build a GDAL VRT mosaicking all chunk waterdepth outputs for one SLR scenario.

A VRT (Virtual Raster Table) is a lightweight XML file that GDAL/rasterio
treats as a single raster covering the union of all source files.  Reading
through the VRT for plotting or area computation is transparent — no data is
copied; GDAL reads the relevant chunk on demand.
"""

from osgeo import gdal

gdal.UseExceptions()

vrt = gdal.BuildVRT(
    str(snakemake.output.waterdepth_vrt),  # noqa: F821
    [str(p) for p in snakemake.input.waterdepth],  # noqa: F821
)
vrt.FlushCache()
del vrt
