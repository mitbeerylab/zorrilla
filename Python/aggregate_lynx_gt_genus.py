import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rdata

data_dir = Path(__file__).parent / ".." / "data"

df = pd.read_csv(data_dir / "bioclip_inference_genus.csv", index_col=0)

# add new column with first path component removed
df["filename"] = ["/".join(filepath.split("/")[1:]) for filepath in df["filepath"]]

# filter for Ain images
df = df[[filepath.split("/")[0] == "pixAin" for filepath in df["filepath"]]]
df["region"] = "ain"

# parse camera trap site ("n_point")
df["n_point"] = [float(os.path.basename(p).split("_")[0]) for p in df["filepath"]]

# load metadata and parse timestamp
metadata = rdata.read_rda(data_dir / "computo-deeplearning-occupany-lynx" / "dat" / "metadata_Ain.RData")["allfiles"]

def parse_dt(dt):
    try:
        return datetime.fromisoformat(dt.replace(":", "-", 2))
    except ValueError:
        return None

metadata["datetime"] = [parse_dt(dt) for dt in metadata["DateTimeOriginal"]]

df = df.merge(metadata[["observed", "pix", "datetime"]], left_on="filename", right_on="pix")



gdf = gpd.read_file(data_dir / "computo-deeplearning-occupany-lynx-shapefiles" / "SIG_01" / "PP01_restreint.shp")

df = df.merge(gdf[["n_point", "lat_Y_Nord", "long_X_Est"]], left_on="n_point", right_on="n_point")

df.to_csv(data_dir / "bioclip_inference_genus_plus_gt.csv")