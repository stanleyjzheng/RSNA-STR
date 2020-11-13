module load cuda/10.2
module load anaconda3/3.7
conda create --name RSNA python=3.7 anaconda
conda activate RSNA 
echo Installing PyTorch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
python -c'import torch; print(torch.cuda.is_available())'
echo Installing requirements.txt
pip3 install -r requirements.txt -q
conda install -c conda-forge pydicom --quiet
conda install gdcm -c conda-forge --quiet