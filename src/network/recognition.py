from .tsn import TSN
# from .tsm import TSM


def build_rec_backbone(rec_string, n_classes, K, modality):
    rec_split = rec_string.split('_')
    algo = rec_split[0]
    backbone = rec_split[1]
    layers = rec_split[2]
    if algo=='tsn':
        model = TSN(n_classes, K, modality, base_model=backbone+layers)
    else:
        raise NotImplementedError

    return model