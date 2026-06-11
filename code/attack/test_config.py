import sys
sys.path.insert(0, '..')
from config import SURROGATE_NAMES, IMAGES_DIR, LABEL_FILE
print('Config loaded successfully')
print(f'Surrogate models: {SURROGATE_NAMES}')
print(f'Images dir: {IMAGES_DIR}')
print(f'Label file: {LABEL_FILE}')
print(f'Images exist: {IMAGES_DIR.exists()}')
print(f'Label exists: {LABEL_FILE.exists()}')
