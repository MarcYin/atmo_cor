mval = [0.2, 0.5, 0.15, 0.15, 0.5, 0.4, 0.3]
for i in range(7):
    x = signal.convolve2d(pre_mod.atom.toa[i], np.ones((3,3)), mode='same').ravel()
    y = pre_mod.atom.boa[i].ravel()
    xy = np.vstack([x,y])
    kde = gaussian_kde(xy)(xy)
    plt.figure(figsize=(5,5))
    plt.scatter(x, y, c=kde, s=4, edgecolor='',\
               norm=colors.LogNorm(vmin=kde.min(), vmax=kde.max()*1.2), cmap = plt.cm.jet,rasterized=True)
    plt.xlim(0,mval[i])
    plt.ylim(0,mval[i])
