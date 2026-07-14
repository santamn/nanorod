# Computational Environment

The architecture of the computer running this program is as follows.

## Hardware

| Item |  |
| :--- | :--- |
| CPU | AMD EPYC 7502P / 32 Core |
| GPU | NVIDIA Tesla A100 40GB x 3, NVIDIA Tesla A100 80GB x 1 |
| Memory | DDR4-3200 32 GB x 8 |
| Storage | Samsung SSD 870 7.28 T |
| Linux Kernel | 5.15.0-97-generic |

## GPUs

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-PCIE-40GB          Off | 00000000:01:00.0 Off |                    0 |
| N/A   27C    P0              32W / 250W |      0MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA A100-PCIE-40GB          Off | 00000000:41:00.0 Off |                    0 |
| N/A   27C    P0              33W / 250W |      0MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA A100-PCIE-40GB          Off | 00000000:81:00.0 Off |                    0 |
| N/A   27C    P0              32W / 250W |      0MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA A100 80GB PCIe          Off | 00000000:C1:00.0 Off |                    0 |
| N/A   30C    P0              43W / 300W |      0MiB / 81920MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
```
