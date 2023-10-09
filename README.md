# Instance Lseg
This repository is the source code for my final year project titled: Instance LSeg: Exploring Instance Information from Visual Language Model.

### Installation
Python Version = 3.7.16
```
conda install ipython
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.4.9
pip install torchmetrics==0.5.1
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install altair
pip install streamlit
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
```

### Data Preparation
By default, for training, testing and demo, we use [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/).

```
python prepare_ade20k.py
```

Note: for demo, if you want to use random inputs, you can ignore data loading and comment the code at [link](https://github.com/isl-org/lang-seg/blob/main/modules/lseg_module.py#L55). 



#### Download Demo LSeg Model
<table>
  <thead>
    <tr style="text-align: right;">
      <th>name</th>
      <th>backbone</th>
      <th>text encoder</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
       <td>Model for demo</td>
      <th>ViT-L/16</th>
      <th>CLIP ViT-B/32</th>
      <td><a href="https://drive.google.com/file/d/1ayk6NXURI_vIPlym16f_RG3ffxBWHxvb/view?usp=sharing">download</a></td>
    </tr>
  </tbody>
</table>



#### Jupyter Notebook
Download the model for demo and put it under folder `checkpoints` as `checkpoints/demo_e200.ckpt`. 

Then run `instance_lseg_demo.ipynb` in your Jupyter Notebook



