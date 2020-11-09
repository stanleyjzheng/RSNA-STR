## A few verified working scripts to be used in Kaggle.

For ease of use, these scripts create their own `config.json` so that a new config does not need to be uploaded for each run.

### Datasets to add:
- [EfficientNet Pytorch 0.7](https://www.kaggle.com/tunguz/efficientnet-pytorch-07)
- [gdcm conda install](https://www.kaggle.com/ronaldokun/gdcm-conda-install)
- [RSNA STR GitHub (a clone of this repo)](www.kaggle.com/dataset/f4127c3bf3b0b540d8d17e1b4f1bddbe4ea05231c9613619e8ccd745c7dd2b17)

### How to install dependencies

For notebooks:
```python
package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
utils_path = '../input/rsna-str-github/'
import sys; sys.path.append(package_path); sys.path.append(utils_path)'
!cp ../input/gdcm-conda-install/gdcm.tar .
!tar -xvzf gdcm.tar
!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2
!cp ../input/rsna-str-github/config.json .
```

For scripts:
```python
package_path = '../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
utils_path = '../input/rsna-str-github/'
import sys; sys.path.append(package_path); sys.path.append(utils_path)

bash_commands = [
            'cp ../input/gdcm-conda-install/gdcm.tar .',
            'tar -xvzf gdcm.tar',
            'conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2',
            'cp ../input/rsna-str-github/config.json .',
            ]
import subprocess
for bashCommand in bash_commands:
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
```

### Shared notebooks:
- [Stage 1 multilabel](https://www.kaggle.com/stanleyjzheng/rsna-github-multilabel-testing?scriptVersionId=46463393)
- [Stage 1 image level](https://www.kaggle.com/stanleyjzheng/rsna-github-image-level-testing?scriptVersionId=46465661)