import gdown
import os

url = 'https://drive.google.com/uc?id=1KvRlk09-F5--Ppm0xHVppUSsVQ2i317n'
output = os.path.join('models', 'rf.pkl')
os.makedirs('models', exist_ok=True)
gdown.download(url, output, quiet=False)
