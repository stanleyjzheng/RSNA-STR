module load anaconda3/3.7

echo Enter your Kaggle username:
read username 
echo Enter your Kaggle API key:
read api_key 

mkdir $HOME/.kaggle
echo '{"username":'$username', "key":'$api_key'}' > $HOME/.kaggle/kaggle.json

echo Installing Kaggle API
pip install kaggle -q

cd input

echo Downloading datasets 
kaggle competitions download -c rsna-str-pulmonary-embolism-detection
mkdir rsna-str-pulmonary-embolism-detection

kaggle datasets download -d khyeh0719/kh-rsna-model
mkdir kh-rsna-model 

echo Downloaded datasets, unzipping datasets

unzip -q rsna-str-pulmonary-embolism-detection.zip rsna-str-pulmonary-embolism-detection
unzip -q kh-rsna-model.zip kh-rsna-model

rm -f *.zip
chmod 600 $HOME/.kaggle