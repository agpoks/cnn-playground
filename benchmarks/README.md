# Benchmarks

Like `sciml-playground`, these eleven architectures don't share one
input/output shape (classification vs. dense per-pixel segmentation vs.
multi-box detection), so benchmarking is grouped by **dataset cluster**.

Every `models/*/example.py` prints one final line in a common format:

```
RESULT: model=<name> metric_name=<name> metric=<value> params=<n> train_time_s=<value>
```

`run_cluster.py` runs every model in a cluster back-to-back with the same
`--device`/`--epochs`, parses that line, and prints a comparison table.

## Clusters

Only `cifar` is a genuine multi-model *comparison* -- the other two are
single-model configs kept in the same format for a consistent interface,
since U-Net (segmentation) and the YOLO-style detector (detection) each
use their own dataset and task, not shared with anything else here.

```bash
python benchmarks/run_cluster.py --cluster cifar --device auto        # AlexNet, VGG, GoogLeNet, ResNet, DenseNet, MobileNet, SE-Net, ViT on CIFAR-10
python benchmarks/run_cluster.py --cluster mnist --device auto        # LeNet-5 on MNIST
python benchmarks/run_cluster.py --cluster segmentation --device auto  # U-Net on Oxford-IIIT Pet
python benchmarks/run_cluster.py --cluster detection --device auto    # YOLO-style detector on Penn-Fudan
```

See `benchmarks/configs/*.yaml` for exactly which models/args each cluster
runs.
