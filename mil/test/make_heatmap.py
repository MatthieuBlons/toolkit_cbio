from mil.visualizer.dissector import HeatmapMaker
from glob import glob
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument("--models_path", type=str, default="/path/with/models")
parser.add_argument(
    "--path_emb",
    type=str,
    help="path the the embeddings = root directory containing mat_pca/ and info/",
    default=None,
)
parser.add_argument(
    "--path_raw",
    type=str,
    help="path to the WSI images in pyramidal format",
    default=None,
)
parser.add_argument(
    "--wsi_ID", type=str, help="name of the WSI to be processed", default=None
)
parser.add_argument(
    "--ds", type=int, help="downsampling factor of the heatmaps", default=16
)
parser.add_argument("--consensus", action="store_true")
args = parser.parse_args()

if args.consensus:
    models = glob(os.path.join(args.models_path, "model_rep_*_best.pt.tar*"))
    if len(models) == 0:
        models = glob(os.path.join(args.models_path, "model_best_test*.pt.tar"))
hm = HeatmapMaker(
    model=args.models_path, path_emb=args.path_emb, path_raw=args.path_raw
)

if args.consensus:
    namedir = "consensus"
else:
    namedir = os.path.basename(args.model).split(".pt.tar")[0]
out = os.path.join(
    os.path.dirname(args.models_path), "summaries_{}".format(namedir), "heatmaps"
)
os.makedirs(out, exist_ok=True)

if args.wsi_ID:
    hm.make_heatmap(args.wsi_ID, out, downsample_factor=args.ds)
else:
    hm.make_all(out, downsample_factor=args.ds)
