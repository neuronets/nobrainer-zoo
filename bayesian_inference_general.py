# imports
import tensorflow as tf
import nobrainer
from nobrainer.volume import standardize_numpy
import nibabel as nib

# Load sample dataset
csv_path = nobrainer.utils.get_data()
filepaths = nobrainer.io.read_csv(csv_path)
# take a sample image and label
feature = filepaths[0]
image_path = feature[0]
# to compare the results with labels
label_path = feature[0]

# image constants
block_shape=(128,128,128)
batch_size = 1

# Load pre-trained model
model_path = "<path_to_trained_model_folder>/trained-models/neuronets/<path_to_bayesian_model>"
model = tf.keras.models.load_model(model_path)

# for bayesian models we cav have variance and entropy outputs
out,variance, entropy = nobrainer.prediction.predict_from_filepath(image_path, 
                                                                   model,
                                                                   block_shape = (128,128,128),
                                                                   batch_size = batch_size,
                                                                   normalizer = standardize_numpy,
                                                                   n_samples=2,
                                                                   return_variance=True,
                                                                   return_entropy=True,
                                                                   )

# Save output
out_file = "out_{}.nii.gz"
nib.save(out, out_file.format("means"))
nib.svae(variance, out_file.format("variance"))
nib.svae(entropy, out_file.format("entropy"))














