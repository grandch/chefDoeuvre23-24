import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import xml.etree.ElementTree as ET
import numpy as np

from sklearn.cluster import KMeans

#--Definition des chemins/variables globales
PARAMOPTI_OUTPUT_PATH = "./"
PARAMOPTI_OUTPUT_FILE = "paramOptiResults.xml"
SHOW_SIGMA = False


#--Definition d'une fonction pour charger les résultats stockés en xml
def load_param_outputs(param_xml_file):
    albedo_r_array, albedo_g_array, albedo_b_array, sigma_t_array, loss_array = [], [], [], [], []
    tree = ET.parse(param_xml_file)
    results = tree.getroot()
    for result in results:
        for final in result.iter("final"):
            for albedo_r in final.iter("albedo_r"):
                albedo_r_array.append(np.float16(albedo_r.text))
            for albedo_g in final.iter("albedo_g"):
                albedo_g_array.append(np.float16(albedo_g.text))
            for albedo_b in final.iter("albedo_b"):
                albedo_b_array.append(np.float16(albedo_b.text))    
            for sigma_t in final.iter("sigma_t"):
                sigma_t_array.append(np.float16(sigma_t.text))
        for loss in result.iter("total_loss"):
            loss_array.append(np.float16(loss.text))
    return np.array(albedo_r_array), np.array(albedo_g_array), np.array(albedo_b_array), np.array(sigma_t_array), np.array(loss_array)


albedo_r_array, albedo_g_array, albedo_b_array, sigma_t_array, loss_array = load_param_outputs(PARAMOPTI_OUTPUT_PATH + PARAMOPTI_OUTPUT_FILE)

#--Quelques statistiques sur les données résultantes
min_loss, max_loss = min(loss_array), max(loss_array)
mean_loss = np.mean(loss_array)
median_loss = np.median(loss_array)
argmin_loss = np.argmin(loss_array)

print(f"Nombre d'échantillons : {len(loss_array)}")
print(f"Valeur moyenne loss : {mean_loss}\nValeur médiane loss : {median_loss}")
print(f"Sachant que la loss est comprise dans l'intervalle [{min_loss}, {max_loss}]")
print(f"Point final minimisant la loss : [{albedo_r_array[argmin_loss]}, {albedo_g_array[argmin_loss]}, {albedo_b_array[argmin_loss]}, {sigma_t_array[argmin_loss]}]")

#Valeurs moyennes de l'albedo/sigma t
mean_albedo_r = np.mean(albedo_r_array)
mean_albedo_g = np.mean(albedo_g_array)
mean_albedo_b = np.mean(albedo_b_array)
mean_sigma_t = np.mean(sigma_t_array)

#Valeurs médianes de l'albedo/sigma t
median_albedo_r = np.median(albedo_r_array)
median_albedo_g = np.median(albedo_g_array)
median_albedo_b = np.median(albedo_b_array)
median_sigma_t = np.median(sigma_t_array)

#Écarts types de l'albedo/sigma t
std_albedo_r = np.std(albedo_r_array)
std_albedo_g = np.std(albedo_g_array)
std_albedo_b = np.std(albedo_b_array)
std_sigma_t = np.std(sigma_t_array)

#--Définition d'une mini-fonction utilitaire
def pretty_print_stats(type_stat, *args):
    print(f"Valeurs {type_stat} : Albedo[{args[0]}, {args[1]}, {args[2]}], Sigma_t[{args[3]}]")

pretty_print_stats("moyennes" ,mean_albedo_r, mean_albedo_g, mean_albedo_b, mean_sigma_t)
pretty_print_stats("médianes", median_albedo_r, median_albedo_g, median_albedo_b, median_sigma_t)
pretty_print_stats("écarts types", std_albedo_r, std_albedo_g, std_albedo_b, std_sigma_t)

#--Algorithme de KMeans pour essayer de faire apparaître des points intéressants en 4D

albedo_sigma_data = np.array([albedo_r_array, albedo_g_array, albedo_b_array, sigma_t_array]).T
filtered_data = albedo_sigma_data[np.where(loss_array<=16860)]
kmeans_4 = KMeans(n_clusters=4, random_state=713705).fit(filtered_data)
kmeans_8 = KMeans(n_clusters=8, random_state=713705).fit(filtered_data)

#--Correlation entre albedo+sigma et loss
print(np.corrcoef(albedo_sigma_data.T, np.array(loss_array)))

#--Visualisation 5D projetée en 3D, coordonées : albedo, couleur : loss, annotation (toggled) : sigma_t 
fig1 = plt.figure()
#fig.tight_layout()
ax11 = fig1.add_subplot(1, 2, 1, projection='3d', xlabel='Albedo Rouge', ylabel='Albedo Vert', zlabel='Albedo Bleu')
ax12 = fig1.add_subplot(1, 2, 2)

normalize = mcolors.Normalize(vmin=min_loss, vmax=max_loss)
colormap = cm.jet
for index in range(len(albedo_r_array)):
    ax11.scatter3D(albedo_r_array[index], albedo_g_array[index], albedo_b_array[index], ".", c=colormap(normalize(loss_array[index])))
    if SHOW_SIGMA:
        ax11.text(albedo_r_array[index], albedo_g_array[index], albedo_b_array[index], str(sigma_t_array[index]), fontsize="xx-small")
ax11.scatter3D(mean_albedo_r, mean_albedo_g, mean_albedo_b, color="black", marker="^", label="Albedo moyen")
ax11.text(mean_albedo_r, mean_albedo_g, mean_albedo_b, str(mean_sigma_t))
ax11.scatter3D(median_albedo_r, median_albedo_g, median_albedo_b, color="black", marker="v", label="Albedo médian")
ax11.text(median_albedo_r, median_albedo_g, median_albedo_b, str(median_sigma_t))
ax11.legend()

scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
scalarmappaple.set_array(loss_array)
plt.colorbar(scalarmappaple, cax=ax12, label='Loss', pad=0.2)

#--Visualisation des points obtenus par KMeans
fig2 = plt.figure()
ax21 = fig2.add_subplot(1, 2, 1, projection='3d', xlabel='Albedo Rouge', ylabel='Albedo Vert', zlabel='Albedo Bleu')
ax22 = fig2.add_subplot(1, 2, 2, projection='3d', xlabel='Albedo Rouge', ylabel='Albedo Vert', zlabel='Albedo Bleu')

for r, g, b, s in kmeans_4.cluster_centers_:
    ax21.scatter3D(r, g, b)
    ax21.text(r, g, b, str(s))
for r, g, b, s in kmeans_8.cluster_centers_:
    ax22.scatter3D(r, g, b)
    ax22.text(r, g, b, str(s))

print(kmeans_4.cluster_centers_)
print(kmeans_8.cluster_centers_)

#--Visualisation des points filtrés
fig3 = plt.figure()
ax31 = fig3.add_subplot(1, 1, 1, projection='3d', xlabel='Albedo Rouge', ylabel='Albedo Vert', zlabel='Albedo Bleu')

for r, g, b, s in filtered_data:
    ax31.scatter3D(r, g, b, color="gray")
    #ax31.text(r, g, b, str(s))

plt.show()