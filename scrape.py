import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, unquote

os.makedirs('txt_files', exist_ok=True)

base_url = 'https://theflameofhope.co/device/CHRISTMAS/'
# base_url = 'https://theflameofhope.co/device/CHILDRENS/'
# base_url = 'https://theflameofhope.co/device/COUNTRY/'
# base_url = 'https://theflameofhope.co/device/GOSPEL/'
# base_url = 'https://theflameofhope.co/device/ROCK/'

response = requests.get(base_url)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')
links = soup.find_all('a', href=lambda href: href and href.endswith('.txt'))

for link in links:
    href = link['href']
    file_url = urljoin(base_url, href)
    
    file_name = os.path.basename(href)
    file_name = unquote(file_name)
    
    print(f'Downloading {file_name}...')
    
    file_response = requests.get(file_url)
    file_response.raise_for_status()
    
    save_path = os.path.join('txt_files', file_name)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(file_response.text)

print('Download complete.')
