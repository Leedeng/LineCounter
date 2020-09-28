import os

dir = os.listdir('.')
train = []
valid = []
for file in dir:
    if file.endswith('train.list'):
        f = os.open(file, os.O_RDWR)
        content = os.read(f, 12)
        print(content)
