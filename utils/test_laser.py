import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('llvm_ad_rgb')

scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'volpath'
    },
    'light1': {
        'type': 'envmap',
        'filename': 'scenes/textures/envmap.exr',
        'scale': 0.1,
    },
    'sphere' : {
        'type': 'sphere',
        'medium': {
            'type': 'homogeneous',
            'albedo': {
                'type': 'rgb',
                'value': [0.0, 0.9, 0.96]
            },
            'sigma_t': 10.,
            'phase': {
                'type': 'hg',
                'g': 0.7
                    }
                }
    },
    'sensor': {
        'type': 'perspective',
        'to_world': mi.ScalarTransform4f.look_at(origin=[4, -4, 0],
                                                 target=[0, 0, 0],
                                                 up=[0, 0, 1]),
        'film': {
		    			'type': 'hdrfilm',
		    			'width': 512,
		    			'height': 512,
		    			'rfilter': {
		        			'type': 'gaussian',
		        			'stddev' : 0.1,
		    			},
		    			'pixel_format': 'rgb',
	       				},
    },
    'light2':{
        'type': 'laser',
        'position': [4,-4,0],
        'radius': 0.1,
        'direction':[-4,4,0],
        'radiance': {
            'type': 'rgb',
            'value': [2, 0.2, 0.4],
        }
    },
    'light3':{
        'type': 'directional',
    'direction': [1.0, 0.0, 0.0],
    'irradiance': {
        'type': 'rgb',
        'value': [1.0, 0.0, 0.0],
    }
    }
})

image = mi.render(scene, spp=16)

plt.imshow(mi.util.convert_to_bitmap(image))
plt.show()
mi.util.write_bitmap('laser_test.png', image)