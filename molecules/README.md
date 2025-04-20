<h2> Pre-training and fine-tuning </h2>

**1. pre-training**

```bash
python chem_tran.py --uniformity_dim <uniformity_dim> --sub_weight <sub_weight>
```

**2. fine-tuning**

```bash
python adv_finetune.py --input_model_file <model_path> --dataset <dataset_name> --uniformity_dim <uniformity_dim> --sub_weight <sub_weight>
```

<h2> reproduced results </h2>

**1. pre-training**
sh scripts/run_pretrain.sh <uniformity_dim> <sub_weight> <device>
**(uniformity_dim, sub_weight)**:
bbbp:(8, 0.8)
tox21:(4, 0.4)
toxcast:(32, 0.4)
sider:(64, 0.2)
clintox:(128, 0.3)
muv:(8, 0.4)
hiv:(4, 0.8)
bace:(16, 0.2)


**2. fine-tuning**
python reproduced.py