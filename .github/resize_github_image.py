import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
img = Image.open('labeled_MultiviewC.png')
H, W, _ = np.array(img).shape
img = img.resize((int(W/10), int(H/10)))
img.save('labeled_MultiviewC.png')
