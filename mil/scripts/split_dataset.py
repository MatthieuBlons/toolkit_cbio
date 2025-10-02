from argparse import ArgumentParser
from mil.deepmil.aide_csv_maker import test_stratif
import os
import datetime

# From a csv table, creates an other csv table with a "test" column, stating in which test fold are each line.
# The created Kfold is stratified with respect to $target_name, a column of the csv.

parser = ArgumentParser()
parser.add_argument("--table", type=str)
parser.add_argument("--target_name", type=str)
parser.add_argument(
    "--equ_vars", type=str, default=None, help="variables to keep in stratif vars"
)
parser.add_argument("-k", type=int, help="number of folds")
parser.add_argument(
    "-rename", type=bool, default=False, help="rename table and save copy"
)
args = parser.parse_args()

target = args.target_name
equ_vars = args.equ_vars
if equ_vars is not None:
    equ_vars = args.equ_vars.split(",")

table = test_stratif(args.table, equ_vars, args.target_name, args.k)

out = args.table
if args.rename:
    date_tag = datetime.date.today().strftime("%Y_%m_%d")
    newname = (
        os.path.splitext(os.path.basename(args.table))[0] + f"_split_{date_tag}.csv"
    )
    out = os.path.join(os.path.dirname(args.table), newname)
print(out)
table.to_csv(out, index=False)
