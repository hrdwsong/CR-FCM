REM =======================CR-DCov=============================
REM ------------------miniImageNet------------------
python meta_train.py --way 5 --shot 1 --dataset miniImageNet --method crdcov --init_weights initialization/miniimagenet/checkpoint1600.pth
python eval.py --way 5 --shot 1 --dataset miniImageNet --method crdcov --save_path results/crdcov-miniImageNet-small-5w1s
python meta_train.py --way 5 --shot 5 --dataset miniImageNet --method crdcov --init_weights initialization/miniimagenet/checkpoint1600.pth
python eval.py --way 5 --shot 5 --dataset miniImageNet --method crdcov --save_path results/crdcov-miniImageNet-small-5w5s

REM ------------------tieredImageNet------------------
python meta_train.py --way 5 --shot 1 --dataset tieredImageNet --method crdcov --init_weights initialization/tieredimagenet/checkpoint0800.pth
python eval.py --way 5 --shot 1 --dataset tieredImageNet --method crdcov --save_path results/crdcov-tieredImageNet-small-5w1s
python meta_train.py --way 5 --shot 5 --dataset tieredImageNet --method crdcov --init_weights initialization/tieredimagenet/checkpoint0800.pth
python eval.py --way 5 --shot 5 --dataset tieredImageNet --method crdcov --save_path results/crdcov-tieredImageNet-small-5w5s

REM ------------------CIFAR-FS------------------
python meta_train.py --way 5 --shot 1 --dataset cifar-fs --method crdcov --init_weights initialization/cifar-fs/checkpoint1600.pth
python eval.py --way 5 --shot 1 --dataset cifar-fs --method crdcov --save_path results/crdcov-cifar-fs-small-5w1s
python meta_train.py --way 5 --shot 5 --dataset cifar-fs --method crdcov --init_weights initialization/cifar-fs/checkpoint1600.pth
python eval.py --way 5 --shot 5 --dataset cifar-fs --method crdcov --save_path results/crdcov-cifar-fs-small-5w5s

REM ------------------FC100------------------
python meta_train.py --way 5 --shot 1 --dataset fc100 --method crdcov --init_weights initialization/fc100/checkpoint1600.pth
python eval.py --way 5 --shot 1 --dataset fc100 --method crdcov --save_path results/crdcov-fc100-small-5w1s
python meta_train.py --way 5 --shot 5 --dataset fc100 --method crdcov --init_weights initialization/fc100/checkpoint1600.pth
python eval.py --way 5 --shot 5 --dataset fc100 --method crdcov --save_path results/crdcov-fc100-small-5w5s

