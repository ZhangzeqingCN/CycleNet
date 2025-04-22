import argparse
from args import Args

from run import run
from layers.Embed import freq_map
from utils.settings import get_setting
from utils.timefeatures import *
from pandas.tseries import offsets


def debug(index):
    freqs = {
        "hd-dw": [HourOfDay, DayOfWeek],
        "hd-hw": [HourOfDay, HourOfWeek],
        "hd-hw-dy": [HourOfDay, HourOfWeek, DayOfYear],
        "hd-hw-hy": [HourOfDay, HourOfWeek, HourOfYear],
    }
    c_name, c_fn = list(freqs.items())[index]
    args = Args(
        model="iTransformer",
        features="M",
        target="OT",
        freq="h",
        seq_len=96,
        label_len=48,
        pred_len=96,
        enc_in=321,
        dec_in=321,
        c_out=321,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=1,
        d_ff=512,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=10,
        itr=1,
        train_epochs=10,
        batch_size=16,
        patience=3,
        learning_rate=0.0005,
        des="test",
        loss="mse",
        lradj="type1",
        use_amp=False,
        use_gpu=True,
        gpu=0,
        use_multi_gpu=True,
        devices="0,1",
        setting_suffix=F"ModifiedTimeFeatures_{c_name}"
    )
    features_by_offsets[offsets.Hour] = c_fn
    freq_map["h"] = len(c_fn)
    run(args, lightning=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, help="index of the feature set to debug")
    debug(ap.parse_args().index)
