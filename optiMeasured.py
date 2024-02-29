import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import rawpy

mi.set_variant('llvm_ad_rgb')

# parsing prg args
parser = argparse.ArgumentParser()
parser.add_argument("-s", help="number of samples for the renders, default is 256", type=int)
parser.add_argument("-rs", help="coeff to resize ref images and renders, default is 0.5", type=float)
args = parser.parse_args()

ref_sample, rs = None, None
if args.s != None:
	ref_sample = args.s
else:
	ref_sample = 128
if args.rs != None:
	rs = args.rs
else:
	rs = 0.5

scene = mi.load_file("scenes/optimizationScene_measured.xml")

def load_sensors_lights_and_refs(sensors_dir):
    # ref picture loading (in folder ./ref/measured)
    dirs = sorted(os.listdir(sensors_dir))
    ref = {}
    for dir in dirs:
        lambdas = os.listdir(sensors_dir + "/" + dir)
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

sensors, laser_dir, ref = load_sensors_lights_and_refs("ref/measured")

params = mi.traverse(scene)
print(params)

for i in range(len(sensors)):
    params['Laser.direction'] = laser_dir[i]
    params.update()
    image = mi.render(scene, spp=ref_sample, sensor=sensors[i])
    # mi.util.write_bitmap('measured' + str(i) + '.png', image)
    plt.imshow(mi.util.convert_to_bitmap(image))
    plt.show()