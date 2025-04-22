from argparse import Namespace

from args import Args
from run import run

if __name__ == '__main__':
    args = Args(model="iTransformerDyT")
    run(args, lightning=True)
