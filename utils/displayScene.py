import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import argparse
import xml.etree.ElementTree as ET
import os

mi.set_variant('scalar_rgb')

# parsing prg args
parser = argparse.ArgumentParser()
parser.add_argument("scene_path", help="path to the scene to render", type=str)
parser.add_argument("sensor_path", help="path to the file describing sensors", type=str)
parser.add_argument("-x", help="x resolution, default is 720", type=int)
parser.add_argument("-y", help="y_resolution, default is 480", type=int)
parser.add_argument("-s", help="number of samples for the renders, default is 512", type=int)
args = parser.parse_args()

ref_sample, xRes, yRes = None, None, None
if args.s != None:
	ref_sample = args.s
else:
	ref_sample = 512

if args.x != None:
	xRes = args.x
else:
	xRes = 720

if args.y != None:
	yRes = args.y
else:
	yRes = 720

# Loading a scene
scene = mi.load_file(args.scene_path)

path_to_sensor = args.sensor_path

def load_sensors(sensors_file):
    origin, target, up = [], [], []
    tree = ET.parse(sensors_file)
    capteurs = tree.getroot()
    for capteur in capteurs:
        o, t, u = [], [], []
        for v_o in capteur.iter('origin'):
            o.append(np.float64(v_o.text))
        for v_t in capteur.iter('target'):
            t.append(np.float64(v_t.text))
        for v_u in capteur.iter('up'):
            u.append(np.float64(v_u.text))
        origin.append(o)
        target.append(t)
        up.append(u)

    origin, target, up = np.array(origin), np.array(target), np.array(up)
    return origin, target, up

origin, target, up = load_sensors(path_to_sensor)

sensor = mi.load_dict({'type': 'perspective',
			'fov_axis' : 'x',
			'fov': 10,
			'to_world': mi.ScalarTransform4f.look_at(
				origin=origin[0],
				target=target[0],
				up=up[0]
			),
			'sampler': {
					'type': 'independent',
					'sample_count': ref_sample
				},
			'film': {
					'type': 'hdrfilm',
					'width': xRes,
					'height': yRes,
					'rfilter': {
						'type': 'gaussian',
						'stddev' : 0.1,
					},
					'pixel_format': 'rgb',
					},
				})

image = mi.render(scene, spp=ref_sample, sensor=sensor)

plt.imshow(mi.util.convert_to_bitmap(image))
plt.show()