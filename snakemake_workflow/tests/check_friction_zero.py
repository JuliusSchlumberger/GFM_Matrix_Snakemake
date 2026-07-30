import rasterio

for t in ["1963", "1981", "1990", "2660", "26729"]:
    with rasterio.open(f"D:/GFM/model_outputs/{t}/inputs/friction.tif") as src:
        f = src.read(1)
    n_zero = int((f == 0).sum())
    n_neg = int((f < 0).sum())
    print(f"{t}: min={f.min():.6f} max={f.max():.4f} n_exact_zero={n_zero:,} "
          f"n_negative={n_neg:,} n_total={f.size:,}")
