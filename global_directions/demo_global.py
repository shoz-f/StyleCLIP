import numpy as np
import clip
from PIL import Image
from MapTS import GetBoundary,GetDt
from manipulate import Manipulator

clip_model,_ = clip.load("ViT-B/32", device = device)

M = Manipulator(dataset_name = 'ffhq')
fs3 = np.load('./npy/ffhq/fs3.npy')

img_index  =   21
img_indexs   = [img_index]
dlatent_tmp  = [tmp[img_indexs] for tmp in M.dlatents]

M.num_images = len(img_indexs)
M.alpha = [0]
M.manipulate_layers = [0]
codes,out = M.EditOneC(0,dlatent_tmp)

original = Image.fromarray(out[0,0]).resize((512,512))
M.manipulate_layers = None
original

neutral = 'smiling face'
target  = 'angry face'
classnames = [target, neutral]
dt = GetDt(classnames, clip_model)
beta   =  0.15
alpha  =  3

M.alpha = [alpha]
boundary_tmp2,c = GetBoundary(fs3, dt, M, threshold = beta)
codes = M.MSCode(dlatent_tmp, boundary_tmp2)
out   = M.GenerateImg(codes)
generated = Image.fromarray(out[0,0]).resize((512,512))
generated
