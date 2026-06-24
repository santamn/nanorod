# コンピュータアーキテクチャ

| 項目 |  |
| :--- | :--- |
| CPU | AMD EPYC 7502P / 32 Core |
| GPU | NVIDIA Tesla A100 40GB x 2, NVIDIA Tesla A100 80GB x 1 |
| Memory | DDR4-3200 32 GB x 8 |
| Storage | Samsung SSD 870 7.28 T |
| Linux Kernel | 5.15.0-97-generic |

## 注意点

- NVIDIA Tesla A100 40GBは一つ壊れており、コンテナ内では使えないようにしている。
- 使用する GPU は `--devices` で指定する（既定は `0,1,2` の3枚）。壊れた A100 が見えない環境で動かすことを前提とする。
