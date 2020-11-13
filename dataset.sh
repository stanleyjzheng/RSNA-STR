pip3 install kaggle -q

echo Enter your Kaggle username:
read username 
echo Enter your Kaggle API key:
read api_key 

mkdir /root/.kaggle
echo {"username":$username, "key":$api_key} > /root/.kaggle/kaggle.json

echo Downloading datasets 
mkdir rsna-str-pulmonary-embolism-detection
kaggle competitions download -c rsna-str-pulmonary-embolism-detection

mkdir kh-rsna-model 
kaggle datasets download -d khyeh0719/kh-rsna-model

echo Downloaded datasets, unzipping datasets

unzip -q rsna-str-pulmonary-embolism-detection.zip rsna-str-pulmonary-embolism-detection
unzip -q kh-rsna-model.zip kh-rsna-model

rm -f /root/.kaggle