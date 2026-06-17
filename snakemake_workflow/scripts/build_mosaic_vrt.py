"""Build GDAL VRTs mosaicking all chunk outputs for one SLR scenario.

A VRT (Virtual Raster Table) is a lightweight XML file that GDAL/rasterio
treats as a single raster covering the union of all source files.  Reading
through the VRT for plotting or area computation is transparent — no data is
copied; GDAL reads the relevant chunk on demand.
"""

from osgeo import gdal

gdal.UseExceptions()

for vrt_path, src_paths in [
    (snakemake.output.flood_count_vrt, list(snakemake.input.flood_count)),  # noqa: F821
    (snakemake.output.waterdepth_vrt, list(snakemake.input.waterdepth)),     # noqa: F821
]:
    vrt = gdal.BuildVRT(str(vrt_path), [str(p) for p in src_paths])
    vrt.FlushCache()
    del vrt
