from dataclasses import dataclass, KW_ONLY
from typing import Optional


@dataclass
class Args:
    _: KW_ONLY
    setting: Optional[str] = None
    setting_prefix: Optional[str] = None
    setting_suffix: Optional[str] = None
    random_seed: int = 2024
    is_training: int = 1
    model_id: str = "Electricity_96_96"
    model: str = "CycleNet"
    data: str = "custom"
    root_path: str = "./dataset/"
    data_path: str = "electricity.csv"
    features: str = "M"
    target: str = "OT"
    freq: str = "h"
    checkpoints: str = "./checkpoints/"
    seq_len: int = 96
    label_len: int = 0
    pred_len: int = 96
    cycle: int = 168
    model_type: str = "linear"
    use_revin: int = 0
    fc_dropout: float = 0.05
    head_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    revin: int = 0
    affine: int = 0
    subtract_last: int = 0
    decomposition: int = 0
    kernel_size: int = 25
    individual: int = 0
    rnn_type: str = "gru"
    dec_way: str = "pmf"
    seg_len: int = 48
    channel_id: int = 1
    period_len: int = 24
    embed_type: int = 0
    enc_in: int = 321
    dec_in: int = 7
    c_out: int = 7
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    moving_avg: int = 25
    factor: int = 1
    distil: bool = True
    dropout: float = 0
    embed: str = "timeF"
    activation: str = "gelu"
    output_attention: bool = False
    do_predict: bool = False
    num_workers: int = 2
    itr: int = 1
    train_epochs: int = 30
    batch_size: int = 16
    patience: int = 5
    learning_rate: float = 0.01
    des: str = "test"
    loss: str = "mse"
    lradj: str = "type3"
    pct_start = 0.3
    use_amp: bool = False
    use_gpu: bool = False
    gpu: int = 0
    use_multi_gpu: bool = False
    devices: str = "01"
    test_flop: bool = False
    dn_layers: int = 3
