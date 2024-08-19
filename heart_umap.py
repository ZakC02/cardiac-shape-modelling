import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from bokeh.plotting import output_notebook, show
import pandas as pd
import torch
import umap
from heart_model import VAE, latent_dim, load_model
from heart_variables import device, model_path
from heart_dataset import heart_dataset, names




vae_model = VAE(latent_dim=latent_dim).to(device)
load_model(vae_model, model_path)



# Get codes of all the training data
vae_model.eval()

codes = []
labels = []

with torch.no_grad():
    for idx in range(len(heart_dataset)):
        oh_image = heart_dataset[idx].unsqueeze(0)
        oh_image = oh_image.to(device)
        code = vae_model.encoder(oh_image)[0].squeeze(0) # Discard logvar
        codes.append(code)
        label = names[idx][names[idx].rfind('/') + 1 : names[idx].find('.nii.gz')]
        labels.append(label.split('_'))

codes = torch.stack(codes).cpu().numpy()
labels = np.array(labels)


reducer = umap.UMAP()
embeddings = reducer.fit_transform(codes)

fig, ax = plt.subplots(figsize = (10, 6))

for group, color in zip(['MINF', 'NOR', 'RV'], ['red', 'green', 'blue']):
    mask = labels[:, 1] == group
    x = embeddings[mask][:, 0]
    y = embeddings[mask][:, 1]
    ax.scatter(x, y, c=color, label=group, s=35, edgecolors='black', linewidths=0.3, alpha=0.7)

ax.legend()
ax.set_title('UMAP projection of the codes by patient category')
plt.show()

fig, ax = plt.subplots(figsize = (10, 8))

ax.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    c=labels[:, 3].astype(int),
    cmap='viridis',
    s=35, edgecolors='black', linewidths=0.3, alpha=0.7
)

sm = ScalarMappable(cmap='viridis')
sm.set_array(labels[:, 3].astype(int))
plt.colorbar(sm, ax=ax, label='Weight (kg)', location='bottom', shrink=0.7, pad=0.08)

ax.set_title('UMAP projection of the codes by patient weight')
plt.show()


mapper = umap.UMAP().fit(codes)

df = pd.DataFrame({
    'Patient number' : labels[:, 0].astype(int),
    'Group' : labels[:, 1],
    'Height' : labels[:, 2].astype(int),
    'Weight' : labels[:, 3].astype(int),
    'ED/ES' : np.where(labels[:, 4].astype(int) == 1, 'ED', 'ES'),
    'Slice index' : labels[:, 5].astype(int)
})

p = umap.plot.interactive(mapper, labels=df['Group'].to_numpy(), hover_data=df)

output_notebook()
show(p)