import argparse
import motmetrics as mm
from collections import OrderedDict
from pathlib import Path
import os, glob
import numpy as np
import pandas as pd
from motmetrics.mot import MOTAccumulator

# source: https://github.com/cheind/py-motmetrics/blob/b68701366b3eee2e41e3273e3a7c9d27fc9cd71a/motmetrics/io.py#L48
def load_motchallenge(fname, **kwargs):
    r"""Load MOT challenge data.

    Params
    ------
    fname : str
        Filename to load data from

    Kwargs
    ------
    sep : str
        Allowed field separators, defaults to '\s+|\t+|,'
    min_confidence : float
        Rows with confidence less than this threshold are removed.
        Defaults to -1. You should set this to 1 when loading
        ground truth MOTChallenge data, so that invalid rectangles in
        the ground truth are not considered during matching.

    Returns
    ------
    df : pandas.DataFrame
        The returned dataframe has the following columns
            'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility'
        The dataframe is indexed by ('FrameId', 'Id')
    """

    sep = kwargs.pop('sep', r'\s+|\t+|,')
    min_confidence = kwargs.pop('min_confidence', -1)
    df = pd.read_csv(
        fname,
        sep=sep,
        index_col=[0, 1],
        skipinitialspace=True,
        header=None,
        names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height', 'Confidence', 'ClassId', 'Visibility', 'unused'],
        engine='python'
    )

    # Account for matlab convention.
    #df[['X', 'Y']] -= (1, 1)

    # Removed trailing column
    del df['unused']

    # Remove all rows without sufficient confidence
    return df[df['Confidence'] >= min_confidence]

def rso_matrix(objs, hyps):
    """Computes includeness matrix between object retangle and hypothesis center.

    The rso_distance is computed as
        rso(a,b) = 1 if (center of b is inside of a)
                 = np.nan (otherwise)

    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
    """

    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asfarray(objs) # Return an array converted to a float type.
    hyps = np.asfarray(hyps)
    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4
    N = objs.shape[0]
    K = hyps.shape[0]
    dist = np.ones((N,K))*np.nan
    for k, hyp in enumerate(hyps) :
        hyp_minx, hyp_miny, w, h = hyp
        hyp_x = hyp_minx + w/2
        hyp_y = hyp_miny + h/2

        for n, obj in enumerate(objs) :
            obj_minx, obj_miny, obj_w, obj_h = obj
            obj_x = obj_minx + w/2
            obj_y = obj_miny + h/2
            if np.abs(hyp_x - obj_x) <= obj_w/2 and np.abs(hyp_y-obj_y) <= obj_h/2 :
                dist[n,k] = 1
                break
    #print (objs, hyps, dist)
    return dist

# source: https://github.com/cheind/py-motmetrics/blob/b68701366b3eee2e41e3273e3a7c9d27fc9cd71a/motmetrics/utils.py#L160
def compare_to_groundtruth(gt, dt, distfields=None):
    """Compare groundtruth and detector results.

    This method assumes both results are given in terms of DataFrames with at least the following fields
     - `FrameId` First level index used for matching ground-truth and test frames.
     - `Id` Secondary level index marking available object / hypothesis ids

    Depending on the distance to be used relevant distfields need to be specified.

    Params
    ------
    gt : pd.DataFrame
        Dataframe for ground-truth
    dt : pd.DataFrame
        Dataframe for detector results

    Kwargs
    ------
    distfields: array, optional
        Fields relevant for extracting distance information. Defaults to ['X', 'Y', 'Width', 'Height']
    """
    # pylint: disable=too-many-locals
    if distfields is None:
        distfields = ['X', 'Y', 'Width', 'Height']

    def compute_rso(a, b):
        return rso_matrix(a, b)

    compute_dist = compute_rso

    acc = MOTAccumulator()

    # We need to account for all frames reported either by ground truth or
    # detector. In case a frame is missing in GT this will lead to FPs, in
    # case a frame is missing in detector results this will lead to FNs.
    allframeids = gt.index.union(dt.index).levels[0]

    gt = gt[distfields]
    dt = dt[distfields]
    fid_to_fgt = dict(iter(gt.groupby('FrameId')))
    fid_to_fdt = dict(iter(dt.groupby('FrameId')))

    for fid in allframeids:
        oids = np.empty(0)
        hids = np.empty(0)
        dists = np.empty((0, 0))
        if fid in fid_to_fgt:
            fgt = fid_to_fgt[fid]
            oids = fgt.index.get_level_values('Id')
        if fid in fid_to_fdt:
            fdt = fid_to_fdt[fid]
            hids = fdt.index.get_level_values('Id')
        if len(oids) > 0 and len(hids) > 0:
            dists = compute_dist(fgt.values, fdt.values)
        acc.update(oids, hids, dists, frameid=fid)

    return acc


def main():
    parser = argparse.ArgumentParser(description='evaluate performace of Detection and Association')
    parser.add_argument('--gt', default='./data/Tracking_GT_TWO_SEQs', help='the directory having test images')
    parser.add_argument('--pred', default='./preds/yolox_nano', help='the directory to save images with bboxes')

    args = parser.parse_args()

    gtfiles = glob.glob(os.path.join(args.gt, '*/gt.txt'))
    tsfiles = glob.glob(os.path.join(args.pred, '*/pred.txt'))
    
    gts = OrderedDict([(Path(f).parts[-2], load_motchallenge(f, min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(Path(f).parts[-2], load_motchallenge(f)) for f in tsfiles])

    mh = mm.metrics.create()

    #accs, names = compare_dataframes(gt, ts)
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            print ('Comparing {}...'.format(k))
            accs.append(compare_to_groundtruth(gts[k], tsacc)) # 0.5
            names.append(k)
        else:
            print ('No ground truth for {}, skipping.'.format(k))

    metrics = ['recall', 'precision', 'num_unique_objects', 'mostly_tracked', \
          'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', \
          'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_objects']
    summary = mh.compute_many(
          accs, names=names,
          metrics=metrics, generate_overall=True)

    div_dict = {
        'num_objects': ['num_false_positives', 'num_misses',
          'num_switches', 'num_fragmentations'],
        'num_unique_objects': ['mostly_tracked', 'partially_tracked',
          'mostly_lost']}
    for divisor in div_dict:
        for divided in div_dict[divisor]:
            summary[divided] = (summary[divided] / summary[divisor])
    fmt = mh.formatters
    change_fmt_list = ['num_false_positives', 'num_misses', 'num_switches',
      'num_fragmentations', 'mostly_tracked', 'partially_tracked',
      'mostly_lost']
    for k in change_fmt_list:
        fmt[k] = fmt['mota']
    print(mm.io.render_summary(
      summary, formatters=fmt,
      namemap=mm.io.motchallenge_metric_names))

    metrics = mm.metrics.motchallenge_metrics + ['num_objects']
    summary = mh.compute_many(
    accs, names=names,
    metrics=metrics, generate_overall=True)
    print(mm.io.render_summary(
        summary, formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names))

if __name__ == '__main__':
    main()            
