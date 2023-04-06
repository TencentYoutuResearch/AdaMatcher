from .resnet_fpn import ResNetFPN

# from .generate_matching_matrix import GenerateMatchingMatrix
# from .feature_interaction import SegmentationModule, FeatureAttention


def build_backbone(config):
    if config["backbone_type"] == "ResNetFPN":
        return ResNetFPN(config["resnetfpn"], config["resolution"])
    else:
        raise ValueError(f"Ada.BACKBONE_TYPE {config['backbone_type']} not supported.")
