import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import argparse
import xml.etree.ElementTree as ET

mi.set_variant('llvm_ad_rgb')


# parsing prg args
parser = argparse.ArgumentParser()
parser.add_argument("scene_path", help="path to the scene to render", type=str)
parser.add_argument("ref_path", help="path to ref images", type=str)
parser.add_argument("-x", help="x resolution, default is 720", type=int)
parser.add_argument("-y", help="y_resolution, default is 480", type=int)
parser.add_argument("-s", help="number of samples for the renders, default is 8", type=int)
parser.add_argument("-fs", help="number of samples for the final renders, default is 4096", type=int)
parser.add_argument("-lr", help="learning rate, default is 0.02", type=float)
parser.add_argument("-it", help="iteration count, default is 40", type=int)
parser.add_argument("-init_sigma_t", help="init value of sigma_t parameter, default is 0.002", type=float)
args = parser.parse_args()

spp, xRes, yRes, lr, iteration_count, fspp, init_sigma_t = None, None, None, None, None, None, None
if args.s != None:
	spp = args.s
else:
	spp = 8
if args.x != None:
	xRes = args.x
else:
	xRes = 720
if args.y != None:
	yRes = args.y
else:
	yRes = 720
if args.lr != None:
    lr = args.lr
else:
    lr = 0.02
if args.it != None:
    iteration_count = args.it
else:
    iteration_count = 40
if args.fs != None:
    fspp = args.fs
else:
    fspp = 4096
if args.init_sigma_t != None:
    init_sigma_t = args.init_sigma_t
else:
    init_sigma_t = 0.002


def img_diff(img1, img2):
    # TODO : check if same size
    npimg1 = np.array(img1)
    npimg2 = np.array(img2)
    diff = np.empty((np.shape(npimg1)[0], np.shape(npimg1)[1]), dtype=float)
    for i in range(np.shape(npimg1)[0]):
        for j in range(np.shape(npimg1)[1]):
            diff[i][j] = np.linalg.norm(npimg1[i][j] - npimg2[i][j])
    return mi.Bitmap(diff)

path_to_ref = args.ref_path
path_to_sensor = args.sensor_path

def load_ref_and_sensors(sensors_file):
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
sensor_count = len(origin)
sensors = []

for i in range(sensor_count):

	sensors.append(mi.load_dict({'type': 'perspective',
				'fov_axis' : 'x',
				'fov': 10,
				'to_world': mi.ScalarTransform4f.look_at(
		    		origin=origin[i],
		    		target=target[i],
		    		up=up[i]
				),
				'sampler': {
		    			'type': 'independent',
		    			'sample_count': spp
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
	    			}))


# LOAD INIT SCENE AND RESAMPLE REF IMAGES

ref_images = [mi.Bitmap(path_to_ref+'synthetic_'+str(i)+'.exr').resample([xRes, yRes]) for i in range(sensor_count)]
# ref_images = [mi.Bitmap(path_to_ref+'synthetic_'+str(i)+'.exr') for i in range(sensor_count)]

scene = mi.load_file(args.scene_path)


# LOAD OPTIMIZER AND INIT AND LOAD OPTIMIZED PARAMETER

params = mi.traverse(scene)

key = 'medium1.sigma_t.value.value'
params[key] = init_sigma_t
params.update()

opt = mi.ad.Adam(lr=lr)
opt[key] = params[key]
params.update(opt)


# OPTIMIZATION ---------------------------------------------

loss_evolution = []

total_loss = 0.0
for it in range(iteration_count):
    total_loss = 0.0
    for sensor_idx in range(sensor_count):
        # Perform the differentiable light transport simulation
        img1 = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it)
        img2 = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it+40)
        
        # Xi Deng L2 loss function
        loss = dr.abs(dr.mean((img1 - ref_images[sensor_idx])*(img2 - ref_images[sensor_idx])))
        

        # classic L2 loss
        # loss = dr.mean(dr.sqr(img1 - ref_images[sensor_idx]))
        
        # Backpropagate gradients
        dr.backward(loss)
        
        # Take a gradient step
        opt.step()
        
        # Clamp the optimized density values. Since we used the `scale` parameter
        # when instantiating the volume, we are in fact optimizing extinction
        # in a range from [1e-6 * scale, scale].
        # opt[key] = dr.clamp(opt[key], 1e-6, 2.0)
        
        # Propagate changes to the scene
        params.update(opt)
        
        total_loss += loss[0]
        print(f"Iteration {it:02d}: Total error={total_loss:6f}, Render {sensor_idx+1:02d}/{sensor_count}: error={loss[0]:6f}", end='\r')
    loss_evolution.append(total_loss)

print("Final total loss="+str(total_loss))

# FINAL RENDER

final_images = []
final_ref = []
final_sub = []

for i in [2,3,5,6,13,31,47]:
    image = mi.render(scene, sensor=sensors[i], spp=fspp)
    final_images.append(image)
    final_ref.append(ref_images[i])

    final_sub.append(img_diff(image, ref_images[i]))

plt.title("Opti sp:"+str(spp)+" Final sp:"+str(fspp)+" Resolution:"+str(xRes)+"*"+str(yRes)+" Nb it:"+str(iteration_count)+" lr:"+str(lr))
plt.subplot(4, 1, 1)
plt.plot(range(iteration_count), loss_evolution, label='Loss evolution')
plt.xlabel("Iteration count")
plt.ylabel("Loss value")


for i in range(len(final_images)):
    plt.subplot(4, 7, 8+i)
    plt.imshow(mi.util.convert_to_bitmap(final_ref[i]))
    plt.xlabel("Sensor "+str(i)+" reference")

    plt.subplot(4, 7, 15+i)
    plt.imshow(mi.util.convert_to_bitmap(final_images[i]))
    plt.xlabel("Sensor "+str(i)+" result")

    plt.subplot(4, 7, 22+i)
    plt.imshow(mi.util.convert_to_bitmap(final_sub[i]))
    plt.xlabel("Sensor "+str(i)+" difference")

plt.show()

print("sigma_t = "+str(params['medium1.sigma_t.value.value']))