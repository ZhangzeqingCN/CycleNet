from argparse import Namespace
import argparse

from args import Args
from run import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dn_layers", type=int)
    dn_layers=ap.parse_args().dn_layers
    args = Args(model="CycleNetReferSeries", dn_layers=dn_layers, setting_suffix=F"dnlayers{dn_layers}")
    run(args, lightning=True)
