# before using this script you must build mitsuba3 in ./ext/mitsuba3/build
# and then perform a 'source ./ext/mitsuba3/build/setpath.sh' in order to use the laser plugin

import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import rawpy

mi.set_variant('llvm_ad_rgb')

# parsing prg args
parser = argparse.ArgumentParser()
parser.add_argument("-s", help="number of samples for the renders, default is 8", type=int)
parser.add_argument("-rs", help="coeff to resize ref images and renders, default is 0.5", type=float)
args = parser.parse_args()

ref_sample, rs = None, None
if args.s != None:
	ref_sample = args.s
else:
	ref_sample = 8
if args.rs != None:
	rs = args.rs
else:
	rs = 0.5

scene = mi.load_file("scenes/optimizationScene_measured.xml")

def img_diff(img1, img2):
    # TODO : check if same size
    npimg1 = np.array(img1)
    npimg2 = np.array(img2)
    diff = np.empty((np.shape(npimg1)[0], np.shape(npimg1)[1]), dtype=float)
    for i in range(np.shape(npimg1)[0]):
        for j in range(np.shape(npimg1)[1]):
            diff[i][j] = np.linalg.norm(npimg1[i][j] - npimg2[i][j])
    return mi.Bitmap(diff)

def load_sensors_lights_and_refs(sensors_dir):
    # ref picture loading (in folder ./ref/measured)
    dirs = sorted(os.listdir(sensors_dir))
    ref = {}
    for dir in dirs:
        lambdas = sorted(os.listdir(sensors_dir + "/" + dir))
        # remove the [0:1] to use all the wave length
        for lam in lambdas[0:1]:
            if ref.__contains__(lam):
                # only the first picture (lower exposure time) is used here
                rgb = rawpy.imread(sensors_dir + "/" + dir + "/" + lam + "/" + sorted(os.listdir(sensors_dir + "/" + dir + "/" + lam))[0]).postprocess()
                bmp = mi.Bitmap(rgb).convert(component_format=mi.Struct.Type.Float32).resample([int(rs * 6744), int(rs * 4502)])
                ref[lam].append(bmp)
            else:
                ref[lam] = []
                # only the first picture (lower exposure time) is used here
                rgb = rawpy.imread(sensors_dir + "/" + dir + "/" + lam + "/" + sorted(os.listdir(sensors_dir + "/" + dir + "/" + lam))[0]).postprocess()
                bmp = mi.Bitmap(rgb).convert(component_format=mi.Struct.Type.Float32).resample([int(rs * 6744), int(rs * 4502)])
                ref[lam].append(bmp)

    # sensors and laser directions construction
    origin, target, up, laser_dir = [], [], [], []

    origin.append([0.4124579436349758*0.839, 0.058085583954402234*0.839, -0.33230771606359427*0.839])
    origin.append([-0.26187965706737165*1.2, 0.0639193905312978*1.2, -0.3371195109510675*1.2])
    origin.append([0.07629510952287223*0.97, 0.06160241046937788*0.97, -0.4743865210065597*0.97])
    origin.append([-0.04723309843906157, 0.06267489306727239, -0.45902214575013506])
    origin.append([0.1995977351808692*0.92, 0.06094895876798736*0.92, -0.4572594416582021*0.92])

    for i in range(5):
        target.append([0,0,0])

    up.append([0.04,-1,0])
    up.append([0,-1,0])
    up.append([0.01,-1,0])
    up.append([-0.03,-1,0])
    up.append([0,-1,0])

    laser_dir.append([0, 0, -1])
    laser_dir.append([0, 0, -1])
    laser_dir.append([0.70710678, 0.0, -0.70710678])
    laser_dir.append([0.5, 0.0, -0.866025])    
    laser_dir.append([0.5, 0.0, -0.866025])

    sensors = []
    for i in range(len(origin)):
        sensors.append(mi.load_dict({'type': 'perspective','focal_length' : '35mm','to_world': mi.ScalarTransform4f.look_at(origin=origin[i],target=target[i],up=up[i]),'sampler': {'type': 'independent','sample_count': ref_sample},'film': {'type': 'hdrfilm','width': int(rs * 6744),'height': int(rs * 4502),'rfilter': {'type': 'gaussian','stddev' : 0.1,},'pixel_format': 'rgb',},}))

    return sensors, laser_dir, ref

sensors, laser_dir, ref_images = load_sensors_lights_and_refs("ref/measured")

# LOAD OPTIMIZER AND INIT AND LOAD OPTIMIZED PARAMETER

params = mi.traverse(scene)

key = 'medium1.sigma_t.value.value'
params[key] = 0.02
params.update()

opt = mi.ad.Adam(lr=0.25)
opt[key] = params[key]
params.update(opt)

# OPTIMIZATION

loss_evolution = []

total_loss = 0.0
for i in range(40):
    total_loss = 0.0
    for sensor in range(len(sensors)):
        params['Laser.direction'] = laser_dir[sensor]
        params.update()

        img1 = mi.render(scene, params, spp=ref_sample, sensor=sensors[sensor], seed=i)
        img2 = mi.render(scene, params, spp=ref_sample, sensor=sensors[sensor], seed=i+200)

        # Xi Deng L2 loss function
        loss = dr.abs(dr.mean((img1 - ref_images['465nm'][0])*(img2 - ref_images['465nm'][0])))

        # Backpropagate gradients
        dr.backward(loss)
        
        # Take a gradient step
        opt.step()

        # Propagate changes to the scene
        params.update(opt)
        
        total_loss += loss[0]
        print(f"Iteration {i:02d}: Total error={total_loss:6f}, Render {sensor+1:02d}/{len(sensors)}: error={loss[0]:6f}", end='\r')
    loss_evolution.append(total_loss)
        

print(total_loss)
plt.plot(range(40), loss_evolution, label='Loss evolution')
plt.show()