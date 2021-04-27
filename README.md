## Introduction
This repository contains a Python implementation for "Multi-modal Machine Learning using Visual Fields and Peripapillary Circular OCT Scans in Detection of Glaucomatous Optic Neuropathy".

<br>
Example of the paired VF-OCT data:
<br>
<p align="center">
   <img src="https://user-images.githubusercontent.com/57675424/116193054-54204780-a761-11eb-86b6-01c7efc9c35a.png"  width="720"/>
</p>

### Setup & Usage for the Code
1. Check dependencies:
   ```shell
   python==3.5.2
   PyTorch==1.4.0
   scipy==1.4.1
   numpy==1.16.4
   ```
   
2. To train the model, you need to setup the corresponding datasets following src/lib/datalayer/oct datalayer.py, then run:
   ```shell
   python src/tools/train.py
   ```
