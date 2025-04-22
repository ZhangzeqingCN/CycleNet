from args import Args
from models import (
    Informer,
    Autoformer,
    Transformer,
    DLinear,
    Linear,
    NLinear,
    PatchTST,
    SegRNN,
    CycleNet,
    LDLinear,
    SparseTSF,
    RLinear,
    RMLP,
    CycleiTransformer,
    iTransformer,
    CycleNetReferSeries,
    iTransformerDyT,
    DiPELinear,
)


def build_model(args: Args):
    model_dict = {
        "Autoformer": Autoformer,
        "Transformer": Transformer,
        "Informer": Informer,
        "DLinear": DLinear,
        "NLinear": NLinear,
        "Linear": Linear,
        "PatchTST": PatchTST,
        "SegRNN": SegRNN,
        "CycleNet": CycleNet,
        "LDLinear": LDLinear,
        "SparseTSF": SparseTSF,
        "RLinear": RLinear,
        "RMLP": RMLP,
        "CycleiTransformer": CycleiTransformer,
        "iTransformer": iTransformer,
        "iTransformerDyT": iTransformerDyT,
        "CycleNetReferSeries": CycleNetReferSeries,
        "DiPELinear": DiPELinear,
    }
    model = model_dict[args.model].Model(args).float()
    return model
