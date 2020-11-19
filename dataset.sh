wget -O rsna-str-pulmonary-embolism-detection.zip "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/22307/1502524/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1605995752&Signature=lRsb3PAOxs8R3epPP4PaX6gQ3Xb%2FP2lSZp7PWSjGisIlyHrFVnDabRJMo6phLafGmMFBY9tqFOSUQwGvFqRu9xldphuwq5lGgvXYUAG0aTq3WjUedyuEzPM4FSEk7UvQJAZ6LceDRbKjfsfSi1hNbFV%2FLKkHfyr22nOrNX4sCRbDWirwMNv8MMyInYl%2FaaD0%2Bb8%2BZCPBX9IifeuM6U3fq9K86ey1zEUVLLNHVXOVQfq9Ow5q%2BIiuYRcu7ydoTv6FDJq0Cu%2BN%2FvlJWZ3L9vKeynCpAeWYxbOhDPJfFPheHOLZ3j%2FVeEMI7mLgiISW4Htt%2FhtpLtzOwcYiVE%2FuhyfoxQ%3D%3D&response-content-disposition=attachment%3B+filename%3Drsna-str-pulmonary-embolism-detection.zip"

mkdir rsna-str-pulmonary-embolism-detection
echo unzipping dataset 

unzip -q rsna-str-pulmonary-embolism-detection.zip -d rsna-str-pulmonary-embolism-detection
rm -f rsna-str-pulmonary-embolism-detection.zip