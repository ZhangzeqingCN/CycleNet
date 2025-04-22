from argparse import Namespace

from args import Args
from run import run

if __name__ == '__main__':
    args = Args(model="iTransformer")
    run(args, lightning=True)
