# cnn sem model

## Getting started

Run training session
The tensorflow model graph and variables will be saved at "tflayers-model" folders by default 

* copy the train.py, predict.py and dataset.py into working folder
* make the training_data, testing_data folders are ready on working folder
* traing_data has all optical and sem image with same size and location
* test _data have same strcture
example see: /gpfs/DEV/FEM/SHARED/MXP_ModelDB/MXP_toolbox/cnn_sem_model

## Run training session

After user put all good and bad images into training_data and testing_data folder, run python command

print help message
```
[qizhang@fdev060601 cnn_sem_model]$ mxp_python train.py -h
usage: train.py [-h] [-t TYPE] [-d DIR] [--input_tag INPUT_TAG]
                [--target_tag TARGET_TAG]

train CNN SEM model

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  train, or model_apply
  -d DIR, --dir DIR     train image folder
  --input_tag INPUT_TAG
                        input image, default is se term image
  --target_tag TARGET_TAG
                        target image, default is sem image
```

use ```python -m py_compile source_code.py``` to compiple pyc file


```
>> ./mxp_python train.pyc --type train --dir ./training_data_3 

Going to read training images
Now going to read sem image files
Now going to read optical image files
Complete reading input data. Will Now print a snippet of it
Number of files in Training-set:                50
Number of files in Validation-set:      49
2018-08-24 16:24:14.502219: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Training Epoch 1 --- Training rms: 0.01947, Validation rms: 0.01937
Training Epoch 2 --- Training rms: 0.01946, Validation rms: 0.01936

...
```

## Run prediction and model_apply

need provide the image path

```
>>./mxp_python predict.pyc --type model_apply
[qizhang@fdev060601 cnn_sem_model]$ mxp_python train.py --type model_apply
Going to read training images
Now going to read input image files
Now going to read target image files
Complete reading input data
Number of files in Training-set:                6
Number of files in Validation-set:      1
Training:    (6, 512, 512, 1) (6, 512, 512, 1)
Validation:  (1, 512, 512, 1) (1, 512, 512, 1)
2018-08-27 12:01:10.833831: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
Loading model from ./tflayers-model/
Saving model apply images in ./model-apply/
Saving model apply images in ./model-apply/
```

It will output the result image on model_apply folder


# how to generate tensorflow model ph file

Suggest use freeze_graph tool from tensorflow package.
For example, you can get it here: /gpfs/DEV/FEM/qizhang/softwares/anaconda2/bin/freeze_graph
You can set alias like this:
```
alias freeze_graph "mxp_python /gpfs/DEV/FEM/qizhang/softwares/anaconda2/lib/python2.7/site-packages/tensorflow/python/tools/freeze_graph.py " 
```

```
freeze_graph \
--input_graph=some_graph_def.pb \
--input_checkpoint=model.ckpt-8361242 \
--output_graph=/tmp/frozen_graph.pb --output_node_names=softmax
```
source code of freeze_graph see here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py



## cnn sem job

### job xml template

refer the lastest job xml teample at : `${branch10}/app/mxp/python/cnn_sem_model/samplejob/job.xml`

```xml
<root>
  <MXP>
    <global>
      <options>
        <enable>0-2100</enable>
      </options>
      <centered_normalize_X>1</centered_normalize_X>
      <device>cpu</device>
    </global>
    <init>
      <data_dir>/gpfs/DEV/FEM/SHARED/MXP_ModelDB/MXP_toolbox/cnn_sem_model/dataset3</data_dir>
      <divide_rule>60:20:20</divide_rule>
      <filter>
        <folder>*</folder>
        <srcfile>*_simulatedSEMImage.pgm</srcfile>
        <tgtfile>*_image.pgm</tgtfile>
      </filter>
      <outxml>mxp_input.xml</outxml>
      <enable>1800</enable>
    </init>
    <DLSEMCalibration>
      <learning_rate>0.001</learning_rate>
      <inxml>mxp_input.xml</inxml>
      <outxml>dlsemcal2000out.xml</outxml>
      <enable>2000</enable>
    </DLSEMCalibration>
    <DLSEMApply>
      <inxml>dlsemcal2000out.xml</inxml>
      <outxml>dlsemapply2100out.xml</outxml>
      <enable>2100</enable>
    </DLSEMApply>
  </MXP>
</root>
```

### global config

```xml
    <global>
      <options>
        <enable>0-2100</enable>
      </options>
      <centered_normalize_X>1</centered_normalize_X>
      <device>cpu</device>
    </global>
```

* **options**: substitution for tachyon job options
    - **enable**: enable range, just like MXP job, you can use this to control which stages to run
* **centered\_normalize_X**: two options
    - `>0`: normalized the source images(X) into array within range of [-1, 1]; 
    - else: raw images
* **device**: 2 options
    - 'cpu', set tensorflow data format into `channels_last`, i.e., dateset array shape as `NHWC`
    - 'gpu', set tensorflow data format into `channels_first`, i.e., dateset array shape as `NCHW`


### init stage

**init** stage for data organization and usage assignment.

```xml
    <init>
      <data_dir>/gpfs/DEV/FEM/SHARED/MXP_ModelDB/MXP_toolbox/cnn_sem_model/dataset3</data_dir>
      <divide_rule>60:20:20</divide_rule>
      <filter>
        <folder>*</folder>
        <srcfile>*_simulatedSEMImage.pgm</srcfile>
        <tgtfile>*_image.pgm</tgtfile>
      </filter>
      <outxml>mxp_input.xml</outxml>
      <enable>1800</enable>
    </init>
```

* **data_dir**: the directory of image dateset, the data hierarchy same as MXP convention, see the example below
    
    ```shell
    [peyang@fdev060501 cnn_sem_model]$ tree /gpfs/DEV/FEM/SHARED/MXP_ModelDB/MXP_toolbox/cnn_sem_model/dataset3
    /gpfs/DEV/FEM/SHARED/MXP_ModelDB/MXP_toolbox/cnn_sem_model/dataset3
    ├── 1
    │   ├── 1_image.pgm
    │   ├── 1_ResistProfileImage.pgm
    │   ├── 1_seTermComponent.pgm
    │   ├── 1_simulatedSEMImage.pgm
    │   └── 1_upsample_ai.pgm
    ...
    └── 530
        ├── 530_image.pgm
        ├── 530_ResistProfileImage.pgm
        ├── 530_seTermComponent.pgm
        ├── 530_simulatedSEMImage.pgm
        └── 530_upsample_ai.pgm
    ```

* **divide\_rule**: by the divide_rule, we divide the dataset patterns into 3 different usages: `training`, `validation`, `test`, dataset in `training` and `validation` will be used in `DLSEMCalibration` stage, dataset in `test` will be used in `DLSEMApply` stage. The 3 division numbers should add up to 100.sub[*]

* **filter**: regex filters for folder and files
    - **folder**: folder regex filter
    - **srcfile**: source file regex filter
    - **tgtfile**: target file regex filter

* **outxml**: as MXP ouxml, but just use depth=1 xml for pattern, this design can greatly simplify our stage I/O code, i.e., the stage inxml and outxml parser become a general parsing process, not necessary to write extra stage specific code. An example as below:

    ```xml
    <root>
        <result>
            <pattern>
                <name>1</name>
                <costwt>1</costwt>
                <usage>training</usage>
                <srcfile>C:\Localdata\D\Note\Python\apps\MXP\cnn_sem_model\dataset\1\1_simulatedSEMImage.pgm</srcfile>
                <tgtfile>C:\Localdata\D\Note\Python\apps\MXP\cnn_sem_model\dataset\1\1_image.pgm</tgtfile>
                <imgpixel>1</imgpixel>
                <offset_x>0</offset_x>
                <offset_y>0</offset_y>
            </pattern>
            ...
        </result>
    </root>
    ```
    
* **enable**: as MXP, enable number for current stage

### DLSEMCalibration stage

**DLSEMCalibration** stage for Deep Learning SEM model calibration

```xml
    <DLSEMCalibration>
      <learning_rate>0.001</learning_rate>
      <inxml>mxp_input.xml</inxml>
      <outxml>dlsemcal2000out.xml</outxml>
      <enable>2000</enable>
    </DLSEMCalibration>
```

There are no special parameters in config currently, please add the parameters on the demand.


### DLSEMApply stage

**DLSEMApply** stage for Deep Learning SEM model apply

```xml
    <DLSEMApply>
      <inxml>dlsemcal2000out.xml</inxml>
      <outxml>dlsemapply2100out.xml</outxml>
      <enable>2100</enable>
    </DLSEMApply>
```

One requirement for inxml in current design, the DL model path should be in its inxml: `root/MXP/result/model`.

### example job results

Run job by command `$ python CnnSemJob.py`

Job results are as below:

```shell
├─data
└─result
    ├─DLSEMApply2100
    ├─DLSEMCalibration2000
    │  ├─logs
    │  └─tflayers-model
    └─init1800

-a---          9/4/2018   6:24 PM        956 dlsemapply2100out.xml             
-a---          9/4/2018   6:24 PM       1001 dlsemcal2000out.xml               
-a---          9/4/2018   6:23 PM        810 mxp_input.xml                     
```