WALT:Watch and Learn 2D Amodal Representation using time-lapse imagery
======================

[N Dinesh Reddy](http://cs.cmu.edu/~dnarapur), [Robert T](http://cs.cmu.edu/~mvo), [Srinivasa G. Narasimhan](http://www.cs.cmu.edu/~srinivas/)

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 

[[Project](https://www.cs.cmu.edu/~walt/)] [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Reddy_WALT_Watch_and_Learn_2D_Amodal_Representation_From_Time-Lapse_Imagery_CVPR_2022_paper.pdf)] [[Supp](https://openaccess.thecvf.com/content/CVPR2022/supplemental/Reddy_WALT_Watch_and_CVPR_2022_supplemental.zip)] [[Bibtex](http://www.cs.cmu.edu/~walt/walt.bib) ]

<img src="vis_cars.gif" width="450" height="350"/><img src="vis_people.gif" width="450" height="350"/>

## Installation

### Setting up with docker

All the stable releases of docker-ce installed from https://docs.docker.com/install/

Install the nvidia-docker from https://github.com/NVIDIA/nvidia-docker

Setting up the docker

```bash
docker build -t walt docker/
```

## Training WaltNet
We Will show the steps to follow to train the walt network to produce amodal segmentation results on any camera in the wild. 

Firstly you need to generate the CWALT data composition. To do that we need to download the walt dataset from [HERE](http://www.cs.cmu.edu/~walt/license.html) and annotations from [HERE](http://www.cs.cmu.edu/~walt/data/annotations.zip)
 
Then CWALT dataset can be generated using 
```bash
python cwalt_generate.py
```

For Training run

```bash
python train.py configs/walt/walt_vehicle.py
```

For Testing run
```bash
python test.py configs/walt/walt_vehicle.py data/models/walt_vehicle.pth
```
