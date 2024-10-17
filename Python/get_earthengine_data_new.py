import ee
import time
import numpy as np
import time
import sys
import json
import argparse
import pandas as pd

ee.Initialize(project='zorrilla')

for image, shortname in [
    (ee.Image("USGS/SRTMGL1_003"), "dem"),
    (ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global").first(), "landcover"),
]:
    nodes = list(set([(row["longitude"], row["latitude"], f"{row['location']:03d}") for _, row in pd.read_csv("data/iwildcam_2022_crops_bioclip_inference_logits_v3.csv").iterrows() if np.isfinite([row["latitude"], row["longitude"]]).all()]))

    # make points from nodes
    points = [ee.Geometry.Point((lon, lat)) for lon, lat, _ in nodes]

    # make features from points (name by list order)
    feats = [ee.Feature(p, {'name': nodes[i][-1]}) for i, p in enumerate(points)]

    # make a featurecollection from points
    fc = ee.FeatureCollection(feats)

    # extract points from DEM
    reducer = ee.Reducer.first()
    data = image.reduceRegions(fc, reducer.setOutputs(['elevation']), 30)

    # see data
    for feat in data.getInfo()['features']:
        print(feat['properties'])

    # export as CSV
    task = ee.batch.Export.table.toDrive(data, 'zorrilla_earthengine', f'iwildcam_{shortname}.csv')
    task.start()