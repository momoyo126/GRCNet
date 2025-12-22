_base_ = ["../_base_/default_runtime.py"]
enable_wandb = False
# misc custom setting
batch_size = 6  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True
seed = 13256645
# weight = 'exp/semantic_kitti/grcnet-w2048-c01/model/model_best.pth'
# model settings
model = dict(
    type="GRCSegmentor",
    backbone=dict(type="GRCNet", nclasses=19, kl=[0.001, 0.001, 0.0001, 0.0001]),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
        )
    ],
)

# scheduler settings
epoch = 50
eval_epoch = 50
# optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
# scheduler = dict(
#     type="OneCycleLR",
#     max_lr=optimizer["lr"],
#     pct_start=0.04,
#     anneal_strategy="cos",
#     div_factor=10.0,
#     final_div_factor=100.0,
# )

optimizer = dict(
    type='SGD', lr=0.24, weight_decay=0.0001, momentum=0.9, nesterov=True)
scheduler = dict(
    type='OneCycleLR',
    max_lr=0.24,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)

# dataset settings
dataset_type = "SemanticKITTIDataset"
data_root = "data/semantic_kitti"
ignore_index = -1
names = [
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
]

sensor = dict(
    name='HDL64',
    type='spherical',
    fov_up=3,
    fov_down= -25,
    img_prop=dict(
        width=1024,
        height=64,
    ),
    img_means=[0,0,0,0,0],
    img_stds=[1,1,1,1,1],
)

data = dict(
    num_classes=19,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5),
            dict(type="AddNoise", p=0.5),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="SemLaserScan",sensor=sensor),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "proj_range_remission"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="SemLaserScan",sensor=sensor),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "proj_range_remission"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            dict(type="SemLaserScan",sensor=sensor),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                return_grid_coord=False,
                return_inverse=True,
                if_not_index_operator=False)
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "proj_range_remission"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
