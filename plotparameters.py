import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

#parameters for plots
plt.rcParams["text.latex.preamble"] = r"\usepackage{txfonts}"
plt.rcParams["font.family"] = 'serif'
plt.rcParams["font.serif"] = 'Times'
plt.rcParams['font.size'] = 13
plt.rcParams["text.usetex"] = True
plt.rcParams['xtick.minor.visible'],plt.rcParams['xtick.top'] = True,True
plt.rcParams['ytick.minor.visible'], plt.rcParams['ytick.right'] = True,True
plt.rcParams['xtick.direction'], plt.rcParams['ytick.direction'] = 'in','in'

#definition of labels examples
Rlabel='$R$ [kpc]'
Zlabel='$Z$ [kpc]'
#Vplabel='$V_\phi^*\,\equiv\,-V_\ell$ [km$\,$s$^{-1}$]' 
#VZlabel='$V_Z^*\,\equiv\,V_b$ [km$\,$s$^{-1}$]' 
#LZlabel='$L_Z^*\,\equiv\,-R\,V_\ell$ [kpc\,km$\,$s$^{-1}$]' 
Vplabel='$V_\phi^*$ [km$\,$s$^{-1}$]' 
VZlabel='$V_Z^*$ [km$\,$s$^{-1}$]' 
LZlabel='$L_Z^*$ [kpc\,km$\,$s$^{-1}$]' 
mLZlabel='$-L_Z^*$ [kpc\,km$\,$s$^{-1}$]' 
RGlabel='$R_g^*$ [kpc]'

sVplabel='$\sigma_\phi^*\,\equiv\,\sigma_\ell$ [km$\,$s$^{-1}$]'
sVZlabel='$\sigma_Z^*\,\equiv\,\sigma_b$ [km$\,$s$^{-1}$]'


def generatecmap(base_map='inferno_r',pixelstofade=25,transparency=False):
    nbOfColours=257
    base = matplotlib.cm.get_cmap(base_map)
    mycolorlist = base(np.linspace(0, 1, nbOfColours))
    mycolorlist[0]=[1,1,1,1]
    incrementsR = 1.*(1 - mycolorlist[pixelstofade][0])/pixelstofade
    incrementsG = 1.*(1 - mycolorlist[pixelstofade][1])/pixelstofade
    incrementsB = 1.*(1 - mycolorlist[pixelstofade][2])/pixelstofade
    for p in range(pixelstofade):
        n = pixelstofade-p
        mycolorlist[p][0] = mycolorlist[pixelstofade][0] + n*incrementsR
        mycolorlist[p][1] = mycolorlist[pixelstofade][1] + n*incrementsG
        mycolorlist[p][2] = mycolorlist[pixelstofade][2] + n*incrementsB
       
    if transparency:
        mycolorlist[:,-1] = np.hstack((np.linspace(1.0,0.0,pixelstofade),
                                       np.ones(nbOfColours-pixelstofade)))

    templatecmap = matplotlib.cm.get_cmap('hot')
    mycolormap = templatecmap.from_list('mycustomcolormap', mycolorlist, nbOfColours)
    #jet hot brg cubehelix etc work)
    return mycolormap


