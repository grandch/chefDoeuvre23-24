import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from random import random
from math import ceil


mi.set_variant('cuda_ad_rgb')

capteur_file = './sensors/sensors.xml'

def img_diff(img1, img2):
    # TODO : check if same size
    npimg1 = np.array(img1)
    npimg2 = np.array(img2)
    diff = np.empty((np.shape(npimg1)[0], np.shape(npimg1)[1]), dtype=float)
    for i in range(np.shape(npimg1)[0]):
        for j in range(np.shape(npimg1)[1]):
            diff[i][j] = np.linalg.norm(npimg1[i][j] - npimg2[i][j])
    return mi.Bitmap(diff)

def load_capt(capt_xml_file):
    origin, target, up = [], [], []
    tree = ET.parse(capteur_file)
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

path_to_ref = "ref/"

ref_spp = 8

origin, target, up = load_capt(capteur_file)
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
		    			'sample_count': ref_spp
					},
				'film': {
		    			'type': 'hdrfilm',
		    			'width': 720,
		    			'height': 480,
		    			'rfilter': {
		        			'type': 'gaussian',
		        			'stddev' : 0.1,
		    			},
		    			'pixel_format': 'rgb',
	       				},
	    			}))


# LOAD INIT SCENE AND REF IMAGES

ref_images = [mi.Bitmap(path_to_ref+'synthetic_'+str(i)+'.exr').resample([720, 480]) for i in range(sensor_count)]

# OPTIMIZATION LOOP ----------------------------

results = []

leap_step = 0.30
step_count = 1
expected_steps = ceil((1/leap_step))**4

sensors_ignored = 41

# 4 parameters
for sig_t in np.arange(0, 1, step=leap_step):
    for alb_r in np.arange(0, 1, step=leap_step):
        for alb_g in np.arange(0, 1, step=leap_step):
            for alb_b in np.arange(0, 1, step=leap_step):

                sig_t_noisy = min(sig_t + random()*leap_step, 1.0)
                alb_r_noisy = min(alb_r + random()*leap_step, 1.0)
                alb_g_noisy = min(alb_g + random()*leap_step, 1.0)
                alb_b_noisy = min(alb_b + random()*leap_step, 1.0)

                print(f'Step {step_count} out of {round(expected_steps)} ({float(step_count/expected_steps*100):.2f}%)')
                print(f'Current param init values : {sig_t_noisy}, [{alb_r_noisy}, {alb_g_noisy}, {alb_b_noisy}]')
                scene = mi.load_file("scenes/optimizationScene.xml")


                # LOAD OPTIMIZER AND LOAD OPTIMIZED PARAMETER

                params = mi.traverse(scene)
                params['medium1.sigma_t.value.value'] = sig_t_noisy
                params['medium1.albedo.value.value'] = [alb_r_noisy, alb_g_noisy, alb_g_noisy]

                keys = [('medium1.sigma_t.value.value', 0.03), ('medium1.albedo.value.value', 0.03)]

                # opt = mi.ad.Adam(lr=0.02)

                opts = []
                for key, lr in keys:
                    opt = mi.ad.Adam(lr=lr)
                    opt[key] = params[key]
                    opts.append(opt)
                    params.update(opt)

                iteration_count = 50
                spp = 8

                # OPTIMIZATION ---------------------------------------------

                loss_evolution = []

                total_loss = 0.0
                for it in range(iteration_count):
                    total_loss = 0.0
                    for sensor_idx in range(sensor_count-sensors_ignored):
                        # Perform the differentiable light transport simulation
                        img1 = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it)
                        img2 = mi.render(scene, params, sensor=sensors[sensor_idx], spp=spp, seed=it+40)
                        
                        # Xi Deng L2 loss function
                        loss = dr.mean((img1 - ref_images[sensor_idx])*(img2 - ref_images[sensor_idx]))

                        # classic L2 loss
                        # loss = dr.mean(dr.sqr(img1 - ref_images[sensor_idx]))
                        
                        # Backpropagate gradients
                        dr.backward(loss)
                        
                        step = int(it%2)
                        # Take a gradient step
                        opts[step].step()
                        
                        # Clamp the optimized density values. Since we used the `scale` parameter
                        # when instantiating the volume, we are in fact optimizing extinction
                        # in a range from [1e-6 * scale, scale].
                        opts[step][keys[step][0]] = dr.clamp(opts[step][keys[step][0]], 1e-6, 2.0)
                        
                        # Propagate changes to the scene
                        params.update(opts[step])

                        
                        total_loss += loss[0]
                        print(f"Iteration {it:02d}: Total error={total_loss:6f}, Render {sensor_idx+1:02d}/{sensor_count-sensors_ignored}: error={loss[0]:6f}", end='\r')
                    loss_evolution.append(total_loss)

                print("Final total loss="+str(total_loss))

                results.append(((sig_t_noisy, alb_r_noisy, alb_g_noisy, alb_b_noisy), params['medium1.sigma_t.value.value'], params['medium1.albedo.value.value'], total_loss))
                step_count += 1
# FINAL RESULTS
            
# <results>
#    <result_k>
#       <init>
#           valeurs initiales
#        </init>
#       <final>
#           valeurs finales
#       </final>
#    </result_k>
# </result>
               
root = ET.Element('results')
for k, res in enumerate(results):
    result = ET.SubElement(root, f'result')

    init = ET.SubElement(result, 'init')
    sigma_t = ET.SubElement(init, 'sigma_t')
    sigma_t.text = str(np.float16(res[0][0]))
    albedo_r = ET.SubElement(init, 'albedo_r')
    albedo_r.text = str(np.float16(res[0][1]))
    albedo_g = ET.SubElement(init, 'albedo_g')
    albedo_g.text = str(np.float16(res[0][2]))
    albedo_b = ET.SubElement(init, 'albedo_b')
    albedo_b.text = str(np.float16(res[0][3]))

    final = ET.SubElement(result, 'final')
    sigma_t = ET.SubElement(final, 'sigma_t')
    sigma_t.text = str(np.float16(res[1][0]))
    albedo_r = ET.SubElement(final, 'albedo_r')
    albedo_r.text = str(np.float16(res[2][0][0]))
    albedo_g = ET.SubElement(final, 'albedo_g')
    albedo_g.text = str(np.float16(res[2][1][0]))
    albedo_b = ET.SubElement(final, 'albedo_b')
    albedo_b.text = str(np.float16(res[2][2][0]))

    total_l = ET.SubElement(result, 'total loss')
    total_l.text = str(np.float16(res[3]))

tree = ET.ElementTree(root)
ET.indent(tree)
tree.write("paramOptiResults.xml")
