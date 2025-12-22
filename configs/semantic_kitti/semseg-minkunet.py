_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 6 # bs: total bs in all gpus
num_worker = 12
mix_prob = 0
empty_cache = False
enable_amp = False
seed = 13256645
# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="MinkUNet",
        nclasses=19,
        cr=0.5
    ),
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
optimizer = dict(type="SGD", lr=2.4e-1, weight_decay=1.0e-4, momentum=0.9, nesterov=True)
# scheduler = dict(
#     type="CosineWarmup",
#     num_epochs=epoch,
#     batch_size=batch_size,
#     dataset_size=19312,
# )
# optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.1,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

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
        width=2048,
        height=64,
    ),
    # img_means=[12.12, 10.88, 0.23, -1.04, 0.21],
    # img_stds=[12.32, 11.47, 6.91, 0.86, 0.16],
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
            # dict(type="AddNoise", p=0.5),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            # dict(type="RandomScale", scale=[0.95, 1.05]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.01, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type='Copy', keys_dict=dict(inverse='origin_inverse')),
            # dict(type="SphereCrop", point_max=80000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=["coord", "strength"],
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
            # dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=False,
            ),
            # dict(type='Copy', keys_dict=dict(inverse='origin_inverse')),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=["coord", "strength"],
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
            # dict(type='Copy', keys_dict=dict(segment='origin_segment')),
            # dict(
            #     type='GridSample',
            #     grid_size=0.05,
            #     hash_type='fnv',
            #     mode='train',
            #     keys=('coord','segment','strength'),
            #     return_inverse=True),
            # dict(type='Copy', keys_dict=dict(inverse='origin_inverse'))
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=False,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=["coord", "strength"],
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
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[1],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
                # [
                #     dict(
                #         type="RandomRotateTargetAngle",
                #         angle=[3 / 2],
                #         axis="z",
                #         center=[0, 0, 0],
                #         p=1,
                #     )
                # ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)