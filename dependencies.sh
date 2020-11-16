module load cuda/11.0
module load anaconda3/3.7
conda create --name RSNA python=3.7 anaconda
conda activate RSNA 
echo Installing PyTorch
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
echo Installing requirements.txt
pip3 install -r requirements.txt -q
conda install -c conda-forge pydicom --quiet
conda install gdcm -c conda-forge --quiet
python3 -c'import torch; print(torch.cuda.is_available())'