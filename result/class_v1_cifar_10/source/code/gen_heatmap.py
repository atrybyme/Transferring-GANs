import numpy as np
import matplotlib.pyplot as plt

corr_coeffs_all = np.load('corr_coeffs.npy')

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2)

labels = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

axes = [ax1, ax2, ax3, ax4]

for ckpt, ax in enumerate(axes):
    corr_coeffs = np.flipud(corr_coeffs_all[ckpt])
    
    im = ax.imshow(corr_coeffs)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels[::-1])

    width, height = corr_coeffs.shape
    for i in range(width):
        for j in range(height):
            text = ax.text(j, i, "{0:.2f}".format(corr_coeffs[i][j]),
                        ha="center", va="center", color="b")

    ax.set_title("{}% Training".format(ckpt/len(axes)*100))
    fig.tight_layout()

plt.show()