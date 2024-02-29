import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import rawpy

mi.set_variant('llvm_ad_rgb')

# parsing prg args
parser = argparse.ArgumentParser()
parser.add_argument("-s", help="number of samples for the renders, default is 128", type=int)
args = parser.parse_args()

ref_sample = None
if args.s != None:
	ref_sample = args.s
else:
	ref_sample = 128

scene = mi.load_file("scenes/optimizationScene_measured.xml")

def load_sensors_lights_and_refs(sensors_dir):
    origin, target, up, laser_dir = [], [], [], []
    dirs = sorted(os.listdir(sensors_dir))
    ref = {}
	
    # for dir in dirs:
    #     lambdas = os.listdir(sensors_dir + "/" + dir)
    #     for lam in lambdas:
    #         if ref.__contains__(lam):
    #             print("i")
    #             ref[lam].append(rawpy.imread(sensors_dir + "/" + dir + "/" + lam + "/" + sorted(os.listdir(sensors_dir + "/" + dir + "/" + lam))[0]))
    #         else:
    #             ref[lam] = []
    #             ref[lam].append(rawpy.imread(sensors_dir + "/" + dir + "/" + lam + "/" + sorted(os.listdir(sensors_dir + "/" + dir + "/" + lam))[0]))

    # origin.append([0.4124579436349758*10, 0.058085583954402234*10, -0.33230771606359427*10])
    # origin.append([-0.26187965706737165*10, 0.0639193905312978*10, -0.3371195109510675*10])
    # origin.append([0.07629510952287223*10, 0.06160241046937788*10, -0.4743865210065597*10])
    # origin.append([-0.04723309843906157*10, 0.06267489306727239*10, -0.45902214575013506*10])
    # origin.append([0.1995977351808692*10, 0.06094895876798736*10, -0.4572594416582021*10])
                
    # origin.append([-0.4124579436349758*5, -0.058085583954402234*5, 0.33230771606359427*5])
    origin.append([0,0,1])
    # origin.append([0.4124579436349758*5, 0.058085583954402234*5, -0.33230771606359427*5])
    origin.append([-0.26187965706737165*5, 0.0639193905312978*5, -0.3371195109510675*5])
    origin.append([0.07629510952287223*5, 0.06160241046937788*5, -0.4743865210065597*5])
    origin.append([-0.04723309843906157*5, 0.06267489306727239*5, -0.45902214575013506*5])
    origin.append([0.1995977351808692*5, 0.06094895876798736*5, -0.4572594416582021*5])

    # target.append([1.11938331,  0.05305775, -1.039578])
    # target.append([-0.96766999,  0.08533974, -1.04521638])
    # target.append([0.07639533,  0.06385106, -1.47438399])
    # target.append([-0.3058195,   0.07056089, -1.42497809])
    # target.append([0.45838115,  0.06012653, -1.42319447])

    for i in range(5):
        target.append([0,0,0])

    up.append([0,1,0])
    up.append([0,1,0])
    up.append([0,1,0])
    up.append([0,1,0])
    up.append([0,1,0])

    laser_dir.append([0, 0, -1])
    laser_dir.append([0, 0, -1])
    laser_dir.append([0.70710678, 0.0, -0.70710678])
    laser_dir.append([0.5, 0.0, -0.866025])    
    laser_dir.append([0.5, 0.0, -0.866025])

    sensors = []
    for i in range(len(origin)):
        sensors.append(mi.load_dict({'type': 'perspective','fov_axis' : 'x','fov': 10,'to_world': mi.ScalarTransform4f.look_at(origin=origin[i],target=target[i],up=up[i]),'sampler': {'type': 'independent','sample_count': ref_sample},'film': {'type': 'hdrfilm','width': 6744,'height': 4502,'rfilter': {'type': 'gaussian','stddev' : 0.1,},'pixel_format': 'rgb',},}))

    return sensors, laser_dir, ref

sensors, laser_dir, ref = load_sensors_lights_and_refs("ref/measured")

params = mi.traverse(scene)
print(params)

for i in range(1):
# for i in range(len(sensors)):
    params['Laser.direction'] = laser_dir[i]
    params.update()
    image = mi.render(scene, spp=ref_sample, sensor=sensors[i])
    plt.imshow(mi.util.convert_to_bitmap(image))
    plt.show()