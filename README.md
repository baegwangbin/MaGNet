
# Multi-View Depth Estimation by Fusing Single-View Depth Probability with Multi-View Geometry

Official implementation of the paper

> **Multi-View Depth Estimation by Fusing Single-View Depth Probability with Multi-View Geometry**
>
> CVPR 2022 [oral] 
>
> [Gwangbin Bae](https://baegwangbin.com), [Ignas Budvytis](https://mi.eng.cam.ac.uk/~ib255/), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/)
>
> [[arXiv]](https://arxiv.org/abs/2112.08177) [[openaccess]](https://openaccess.thecvf.com/content/CVPR2022/html/Bae_Multi-View_Depth_Estimation_by_Fusing_Single-View_Depth_Probability_With_Multi-View_CVPR_2022_paper.html) [[oral presentation]](https://www.youtube.com/watch?v=LM113ibJVmQ)

<p align="center">
  <img width=100% src="https://github.com/baegwangbin/MaGNet/blob/master/figs/method.png?raw=true?raw=true">
</p>

*We present **MaGNet** (**M**onocular **a**nd **G**eometric **Net**work), a novel framework for fusing single-view depth probability with multi-view geometry, to improve the accuracy, robustness and efficiency of multi-view depth estimation. For each frame, MaGNet estimates a single-view depth probability distribution, parameterized as a pixel-wise Gaussian. The distribution estimated for the reference frame is then used to sample per-pixel depth candidates. Such probabilistic sampling enables the network to achieve higher accuracy while evaluating fewer depth candidates. We also propose depth consistency weighting for the multi-view matching score, to ensure that the multi-view depth is consistent with the single-view predictions. The proposed method achieves state-of-the-art performance on ScanNet, 7-Scenes and KITTI. Qualitative evaluation demonstrates that our method is more robust against challenging artifacts such as texture-less/reflective surfaces and moving objects.*


## Datasets

We evaluated MaGNet on ScanNet, 7-Scenes and KITTI

### ScanNet

* In order to download ScanNet, you should submit an agreement to the Terms of Use. Please follow the instructions in [this link](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation).
* The folder should be organized as

>`/path/to/ScanNet` \
>`/path/to/ScanNet/scans` \
>`/path/to/ScanNet/scans/scene0000_00 ...` \
>`/path/to/ScanNet/scans_test` \
>`/path/to/ScanNet/scans_test/scene0707_00 ...` 

### 7-Scenes

* Download all seven scenes (Chess, Fire, Heads, Office, Pumpkin, RedKitchen, Stairs) from [this link](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/).
* The folder should be organized as:

>`/path/to/SevenScenes` \
>`/path/to/SevenScenes/chess ...` 

### KITTI 

* Download raw data from [this link](https://www.cvlibs.net/datasets/kitti/raw_data.php).
* Download depth maps from [this link](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php)
* The folder should be organized as:

>`/path/to/KITTI` \
>`/path/to/KITTI/rawdata` \
>`/path/to/KITTI/rawdata/2011_09_26 ...` \
>`/path/to/KITTI/train` \
>`/path/to/KITTI/train/2011_09_26_drive_0001_sync ...` \
>`/path/to/KITTI/val` \
>`/path/to/KITTI/val/2011_09_26_drive_0002_sync ...`

## Download model weights

Download model weights by

```python
python ckpts/download.py
```

If some files are not downloaded properly, download them manually from [this link](https://drive.google.com/drive/u/0/folders/1O4yng6XFe8Wy2fHZp_xjO9KDTfYE7yDB) and place the files under `./ckpts`.

## Install dependencies

We recommend using a virtual environment.
```
python3.6 -m venv --system-site-packages ./venv
source ./venv/bin/activate
```

Install the necessary dependencies by
```
python3.6 -m pip install -r requirements.txt
```

## Test scripts

If you wish to evaluate the accuracy of our D-Net (single-view), run

```python
python test_DNet.py ./test_scripts/dnet/scannet.txt
python test_DNet.py ./test_scripts/dnet/7scenes.txt
python test_DNet.py ./test_scripts/dnet/kitti_eigen.txt
python test_DNet.py ./test_scripts/dnet/kitti_official.txt
```

You should get the following results:

|Dataset|abs_rel|abs_diff|sq_rel|rmse|rmse_log|irmse|log_10|silog|a1|a2|a3|NLL|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|ScanNet|0.1186|0.2070|0.0493|0.2708|0.1461|0.1086|0.0515|10.0098|0.8546|0.9703|0.9928|2.2352|
|7-Scenes|0.1339|0.2209|0.0549|0.2932|0.1677|0.1165|0.0566|12.8807|0.8308|0.9716|0.9948|2.7941|
|KITTI (eigen)|0.0605|1.1331|0.2086|2.4215|0.0921|0.0075|0.0261|8.4312|0.9602|0.9946|0.9989|2.6443|
|KITTI (official)|0.0629|1.1682|0.2541|2.4708|0.1021|0.0080|0.0270|9.5752|0.9581|0.9905|0.9971|1.7810|

In order to evaluate the accuracy of the full pipeline (multi-view), run

```python
python test_MaGNet.py ./test_scripts/magnet/scannet.txt
python test_MaGNet.py ./test_scripts/magnet/7scenes.txt
python test_MaGNet.py ./test_scripts/magnet/kitti_eigen.txt
python test_MaGNet.py ./test_scripts/magnet/kitti_official.txt
```

You should get the following results:

|Dataset|abs_rel|abs_diff|sq_rel|rmse|rmse_log|irmse|log_10|silog|a1|a2|a3|NLL|
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|ScanNet|0.0810|0.1466|0.0302|0.2098|0.1101|0.1055|0.0351|8.7686|0.9298|0.9835|0.9946|0.1454|
|7-Scenes|0.1257|0.2133|0.0552|0.2957|0.1639|0.1782|0.0527|13.6210|0.8552|0.9715|0.9935|1.5605|
|KITTI (eigen)|0.0535|0.9995|0.1623|2.1584|0.0826|0.0566|0.0235|7.4645|0.9714|0.9958|0.9990|1.8053|
|KITTI (official)|0.0503|0.9135|0.1667|1.9707|0.0848|0.2423|0.0219|7.9451|0.9769|0.9941|0.9979|1.4750|

## Training scripts

Coming soon


## Citation

If you find our work useful in your research please consider citing our paper:

```
@InProceedings{Bae2022,
  title = {Multi-View Depth Estimation by Fusing Single-View Depth Probability with Multi-View Geometry}
  author = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
  booktitle = {Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}                         
}
```
