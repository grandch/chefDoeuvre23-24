import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

def img_diff(img1, img2):
    # TODO : check if same size
    npimg1 = np.array(img1)
    npimg2 = np.array(img2)
    diff = np.empty((np.shape(npimg1)[0], np.shape(npimg1)[1]), dtype=float)
    for i in range(np.shape(npimg1)[0]):
        for j in range(np.shape(npimg1)[1]):
            diff[i][j] = np.linalg.norm(npimg1[i][j] - npimg2[i][j])
    return mi.Bitmap(diff)

path_to_ref = "ref/"

ref_spp = 8

# Geometry 0
origin0 = [2.8350489, 98.868300, 0.9857774]
target0 = [2.8058696359579196, 97.86884346912309, 0.9704662388713631]
up0 = [0.99784, 0.02822, -0.05922]

# Geometry 1
origin1 = [5.914261520165206, 70.50714494081035, 66.43324240065756]
target1 = [5.852102807991802, 69.8124070184323, 65.71667034307313]
up1 = [0.9976003115283925, -0.021311238802965066, -0.06587449839761081]

# Geometry 2
origin2 = [6.229279776476691, 4.1309519173533555, 92.72600026085522]
target2 = [6.163787505555136, 4.14801450389434, 91.72829307405637]
up2 = [0.9977707916714061, -0.011720348491981872, -0.06569688515174432]

# Geometry 3
origin3 = [-61.41774646598098, 4.290400392339096, 69.76857824439736]
target3 = [-60.75854004824276, 4.305754903894482, 69.01677299183017]
up3 = [0.751961924180018, -0.012923062561064994, 0.6590798578605839]

# Geometry 4
origin4 = [-93.0178805232148, 4.4825067794940425, 5.70155527932938]
target4 = [-92.02011568910623, 4.495803283107566, 5.636068343166717]
up4 = [0.06565429110156082, -0.012565779247196672, 0.9977633062264124]

# Geometry 5
origin5 = [6.229279776476691, 4.1309519173533555, 92.72600026085522]
target5 = [6.163787505555136, 4.14801450389434, 91.72829307405637]
up5 = [0.9977707916714061, -0.011720348491981872, -0.06569688515174432]

# Geometry 6
origin6 = [-93.0178805232148, 4.4825067794940425, 5.70155527932938]
target6 = [-92.02011568910623, 4.495803283107566, 5.636068343166717]
up6 = [0.06565429110156082, -0.012565779247196672, 0.9977633062264124]

# Geometry 7
origin7 = [-70.06019243730509, 4.594737762641323, -61.94547548228039]
target7 = [-69.30833238776984, 4.606831878497592, -61.28626368151034]
up7 = [-0.6591224040667827, -0.010857790270031366, 0.7519572892443269]

# Geometry 8
origin8 = [-5.992984528246556, 4.561349953971726, -93.54580087455095]
target8 = [-5.9274447764776275, 4.573801644696316, -92.54802860778054]
up8 = [-0.9978038029873474, -0.008799612408008762, 0.06565163794952175]

# Geometry 9
origin9 = [61.65404171421115, 4.4019014789859865, -70.5883788580931]
target9 = [60.99488277732029, 4.4160612446961744, -69.83650852555434]
up9 = [-0.7519949354959592, -0.0075968983389256405, -0.6591251050628066]

# Geometry 10
origin10 = [93.25417577144495, 4.20979509183104, -6.521355893025131]
target10 = [92.25645841818377, 4.226012865483091, -6.4558038768909025]
up10 = [-0.06568730241750223, -0.007954181652793962, -0.9978085534286352]

# Geometry 11
origin11 = [70.29648768553525, 4.09756410868376, 61.12567486858465]
target11 = [69.54467511684737, 4.114984270093066, 60.466528147786164]
up11 = [0.6590893927508412, -0.009662170629959268, -0.7520025364465498]

# Geometry 12
origin12 = [58.91471146831826, 57.863086834181026, 49.855693674866366]
target12 = [58.28447525596191, 57.3039965093412, 49.3169726660824]
up12 = [0.65841598727409, -0.017157000826265357, -0.7524586533654408]

# Geometry 13
origin13 = [77.17847683369592, 57.957632828621186, -6.346478238462177]
target13 = [76.35250792886144, 57.39752900462007, -6.282767371368293]
up13 = [-0.06647784075297751, -0.015450138481060548, -0.9976682764875996]

# Geometry 14
origin14 = [50.351972919800474, 58.116896255497146, -59.00169179762058]
target14 = [49.81358375885936, 57.55508530011452, -58.373595020210296]
up14 = [-0.7524465849418178, -0.015094512875064385, -0.6584802901305854]

# Geometry 15
origin15 = [-5.850198111663067, 58.247582759334975, -77.26523702930555]
target15 = [-5.786155406048276, 57.68437105476116, -76.44140991899138]
up15 = [-0.9976590583106485, -0.016298444665142273, 0.06641358349168337]

# Geometry 16
origin16 = [-58.505566705275626, 58.2731379586056, -50.43857683381053]
target16 = [-57.877136427465096, 57.70965242690962, -49.90233114245229]
up16 = [-0.6584731196230601, -0.018356686936838704, 0.75238034449245]

# Geometry 17
origin17 = [-76.7693320706533, 58.17859196416545, 5.763595079518001]
target17 = [-75.94516910036465, 57.616119931630756, 5.6974088949983885]
up17 = [0.06642070840400739, -0.020063549282043514, 0.9975899676146088]

# Geometry 18
origin18 = [6.259342874705679, 57.88864203345165, 76.68235387036141]
target18 = [6.193494234545066, 57.329277881489666, 75.85605144262152]
up18 = [0.9976019259616785, -0.019215243097961788, -0.06649189236467391]

# Geometry 19
origin19 = [6.229279776476691, 4.1309519173533555, 92.72600026085522]
target19 = [6.163787505555136, 4.14801450389434, 91.72829307405637]
up19 = [0.9977707916714061, -0.011720348491981872, -0.06569688515174432]

# Geometry 20
origin20 = [-93.0178805232148, 4.4825067794940425, 5.70155527932938]
target20 = [-92.02011568910623, 4.495803283107566, 5.636068343166717]
up20 = [0.06565429110156082, -0.012565779247196672, 0.9977633062264124]

# Geometry 21
origin21 = [-70.06019243730509, 4.594737762641323, -61.94547548228039]
target21 = [-69.30833238776984, 4.606831878497592, -61.28626368151034]
up21 = [-0.6591224040667827, -0.010857790270031366, 0.7519572892443269]

# Geometry 22
origin22 = [-5.992984528246556, 4.561349953971726, -93.54580087455095]
target22 = [-5.9274447764776275, 4.573801644696316, -92.54802860778054]
up22 = [-0.9978038029873474, -0.008799612408008762, 0.06565163794952175]

# Geometry 23
origin23 = [61.65404171421115, 4.4019014789859865, -70.5883788580931]
target23 = [60.99488277732029, 4.4160612446961744, -69.83650852555434]
up23 = [-0.7519949354959592, -0.0075968983389256405, -0.6591251050628066]

# Geometry 24
origin24 = [93.25417577144495, 4.20979509183104, -6.521355893025131]
target24 = [92.25645841818377, 4.226012865483091, -6.4558038768909025]
up24 = [-0.06568730241750223, -0.007954181652793962, -0.9978085534286352]

# Geometry 25
origin25 = [70.29648768553525, 4.09756410868376, 61.12567486858465]
target25 = [69.54467511684737, 4.114984270093066, 60.466528147786164]
up25 = [0.6590893927508412, -0.009662170629959268, -0.7520025364465498]

# Geometry 26
origin26 = [58.91471146831826, 57.863086834181026, 49.855693674866366]
target26 = [58.28447525596191, 57.3039965093412, 49.3169726660824]
up26 = [0.65841598727409, -0.017157000826265357, -0.7524586533654408]

# Geometry 27
origin27 = [77.17847683369592, 57.957632828621186, -6.346478238462177]
target27 = [76.35250792886144, 57.39752900462007, -6.282767371368293]
up27 = [-0.06647784075297751, -0.015450138481060548, -0.9976682764875996]

# Geometry 28
origin28 = [50.351972919800474, 58.116896255497146, -59.00169179762058]
target28 = [49.81358375885936, 57.55508530011452, -58.373595020210296]
up28 = [-0.7524465849418178, -0.015094512875064385, -0.6584802901305854]

# Geometry 29
origin29 = [-5.850198111663067, 58.247582759334975, -77.26523702930555]
target29 = [-5.786155406048276, 57.68437105476116, -76.44140991899138]
up29 = [-0.9976590583106485, -0.016298444665142273, 0.06641358349168337]

# Geometry 30
origin30 = [-58.505566705275626, 58.2731379586056, -50.43857683381053]
target30 = [-57.877136427465096, 57.70965242690962, -49.90233114245229]
up30 = [-0.6584731196230601, -0.018356686936838704, 0.75238034449245]

# Geometry 31
origin31 = [-76.7693320706533, 58.17859196416545, 5.763595079518001]
target31 = [-75.94516910036465, 57.616119931630756, 5.6974088949983885]
up31 = [0.06642070840400739, -0.020063549282043514, 0.9975899676146088]

# Geometry 32
origin32 = [6.259342874705679, 57.88864203345165, 76.68235387036141]
target32 = [6.193494234545066, 57.329277881489666, 75.85605144262152]
up32 = [0.9976019259616785, -0.019215243097961788, -0.06649189236467391]

# Geometry 33
origin33 = [6.229279776476691, 4.1309519173533555, 92.72600026085522]
target33 = [6.163787505555136, 4.14801450389434, 91.72829307405637]
up33 = [0.9977707916714061, -0.011720348491981872, -0.06569688515174432]

# Geometry 34
origin34 = [-93.0178805232148, 4.4825067794940425, 5.70155527932938]
target34 = [-92.02011568910623, 4.495803283107566, 5.636068343166717]
up34 = [0.06565429110156082, -0.012565779247196672, 0.9977633062264124]

# Geometry 35
origin35 = [-70.06019243730509, 4.594737762641323, -61.94547548228039]
target35 = [-69.30833238776984, 4.606831878497592, -61.28626368151034]
up35 = [-0.6591224040667827, -0.010857790270031366, 0.7519572892443269]

# Geometry 36
origin36 = [-5.992984528246556, 4.561349953971726, -93.54580087455095]
target36 = [-5.9274447764776275, 4.573801644696316, -92.54802860778054]
up36 = [-0.9978038029873474, -0.008799612408008762, 0.06565163794952175]

# Geometry 37
origin37 = [61.65404171421115, 4.4019014789859865, -70.5883788580931]
target37 = [60.99488277732029, 4.4160612446961744, -69.83650852555434]
up37 = [-0.7519949354959592, -0.0075968983389256405, -0.6591251050628066]

# Geometry 38
origin38 = [93.25417577144495, 4.20979509183104, -6.521355893025131]
target38 = [92.25645841818377, 4.226012865483091, -6.4558038768909025]
up38 = [-0.06568730241750223, -0.007954181652793962, -0.9978085534286352]

# Geometry 39
origin39 = [70.29648768553525, 4.09756410868376, 61.12567486858465]
target39 = [69.54467511684737, 4.114984270093066, 60.466528147786164]
up39 = [0.6590893927508412, -0.009662170629959268, -0.7520025364465498]

# Geometry 40
origin40 = [58.91471146831826, 57.863086834181026, 49.855693674866366]
target40 = [58.28447525596191, 57.3039965093412, 49.3169726660824]
up40 = [0.65841598727409, -0.017157000826265357, -0.7524586533654408]

# Geometry 41
origin41 = [77.17847683369592, 57.957632828621186, -6.346478238462177]
target41 = [76.35250792886144, 57.39752900462007, -6.282767371368293]
up41 = [0.06647784075297751, -0.015450138481060548, -0.9976682764875996]

# Geometry 42
origin42 = [50.351972919800474, 58.116896255497146, -59.00169179762058]
target42 = [49.81358375885936, 57.55508530011452, -58.373595020210296]
up42 = [-0.7524465849418178, -0.015094512875064385, -0.6584802901305854]

# Geometry 43
origin43 = [-5.850198111663067, 58.247582759334975, -77.26523702930555]
target43 = [-5.786155406048276, 57.68437105476116, -76.44140991899138]
up43 = [-0.9976590583106485, -0.016298444665142273, 0.06641358349168337]

# Geometry 44
origin44= [-58.505566705275626, 58.2731379586056, -50.43857683381053]
target44= [-57.877136427465096, 57.70965242690962, -49.90233114245229]
up44 = [-0.6584731196230601, -0.018356686936838704, 0.75238034449245]

# Geometry 45
origin45 = [-76.7693320706533, 58.17859196416545, 5.763595079518001]
target45 = [-75.94516910036465, 57.616119931630756, 5.6974088949983885]
up45 = [0.06642070840400739, -0.020063549282043514, 0.9975899676146088]

# Geometry 46
origin46 = [6.259342874705679, 57.88864203345165, 76.68235387036141]
target46 = [6.193494234545066, 57.329277881489666, 75.85605144262152]
up46 = [0.9976019259616785, -0.019215243097961788, -0.06649189236467391]

# Geometry 47
origin47 = [4.457829895703712, 92.73787341619244, 32.755433161868545]
target47 = [4.411264435971933, 91.80447230869792, 32.39963261312253]
up47 = [0.9976890850250391, -0.025815216146589, -0.06284953649156828]

# Geometry 48
origin48 = [4.457829895703712, 92.73787341619244, 32.755433161868545]
target48 = [4.411264435971933, 91.80447230869792, 32.39963261312253]
up48 = [0.9976890850250391, -0.025815216146589, -0.06284953649156828]

# Geometry 49
origin49 = [4.457829895703712, 92.73787341619244, 32.755433161868545]
target49 = [4.411264435971933, 91.80447230869792, 32.39963261312253]
up49 = [0.9976890850250391, -0.025815216146589, -0.06284953649156828]

# Geometry 50
origin50 = [4.457829895703712, 92.73787341619244, 32.755433161868545]
target50 = [4.411264435971933, 91.80447230869792, 32.39963261312253]
up50 = [0.9976890850250391, -0.025815216146589, -0.06284953649156828]

origin = np.array([origin0, origin1, origin2, origin3, origin4, origin5, origin6, origin7, origin8, origin9, origin10,origin11, origin12, origin13, origin14, origin15, origin16, origin17, origin18, origin19, origin20, origin21, origin22, origin23, origin24, origin25, origin26, origin27, origin28, origin29, origin30, origin31, origin32, origin33, origin34, origin35, origin36, origin37, origin38,origin39, origin40,origin41,origin42,origin43,origin44, origin45, origin46, origin47, origin48,origin49, origin50])

target = np.array([target0, target1, target2, target3, target4, target5, target6, target7, target8, target9, target10,target11, target12, target13, target14, target15, target16, target17, target18, target19, target20, target21, target22, target23, target24, target25, target26, target27, target28, target29, target30, target31, target32, target33, target34, target35, target36, target37, target38,target39, target40,target41,target42,target43,target44, target45, target46, target47, target48,target49, target50])

up = np.array([up0, up1, up2, up3, up4, up5, up6, up7, up8, up9, up10,up11, up12, up13, up14, up15, up16, up17, up18, up19, up20, up21, up22, up23, up24, up25, up26, up27, up28, up29, up30, up31, up32, up33, up34, up35, up36, up37, up38,up39, up40,up41,up42,up43,up44, up45, up46, up47, up48,up49, up50])

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

scene = mi.load_file("scenes/optimizationScene.xml")


# LOAD OPTIMIZER AND LOAD OPTIMIZED PARAMETER

params = mi.traverse(scene)

key = 'medium1.sigma_t.value.value'

opt = mi.ad.Adam(lr=0.02)
opt[key] = params[key]
params.update(opt)

iteration_count = 20
spp = 4

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
        loss = dr.mean((img1 - ref_images[sensor_idx])*(img2 - ref_images[sensor_idx]))
        
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
    image = mi.render(scene, sensor=sensors[i], spp=ref_spp)
    final_images.append(image)
    final_ref.append(ref_images[i])

    final_sub.append(img_diff(image, ref_images[i]))

plt.subplot(4, 1, 1)
plt.plot(range(iteration_count), loss_evolution, label='Loss evolution')


for i in range(len(final_images)):
    plt.subplot(4, 7, 8+i)
    plt.imshow(mi.util.convert_to_bitmap(final_ref[i]))

    plt.subplot(4, 7, 15+i)
    plt.imshow(mi.util.convert_to_bitmap(final_images[i]))

    plt.subplot(4, 7, 22+i)
    plt.imshow(mi.util.convert_to_bitmap(final_sub[i]))

plt.show()

print("sigma_t = "+str(params['medium1.sigma_t.value.value']))