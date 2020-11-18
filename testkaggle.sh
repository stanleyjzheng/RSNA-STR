module load anaconda3/3.7

echo Enter your Kaggle username:
read username 
echo Enter your Kaggle API key:
read api_key 

mkdir $HOME/.kaggle
echo '{"username":'$username', "key":'$api_key'}' > $HOME/.kaggle/kaggle.json

echo Installing Kaggle API
pip install kaggle -q

chmod 600 $HOME/.kaggle/kaggle.json 

echo Testing Kaggle, you should see a list of active competitions...
kaggle competitions list