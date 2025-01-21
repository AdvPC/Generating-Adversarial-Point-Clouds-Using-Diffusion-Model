# **Usage Instructions**

## **Overview**
This code is designed to process and generate 3D point clouds, create adversarial samples, and calculate distances between generated point clouds and reference point clouds. The main functions of the code include:
1. Generating point clouds using a pretrained FlowVAE model.
2. Using the PointNet classification model to classify generated point clouds, apply gradient optimization, and generate adversarial samples.
3. Calculating various distance metrics (Chamfer Distance, Hausdorff Distance, and MSE) between generated point clouds and reference point clouds.
4. Saving the generated point clouds and reference point clouds as `.npy` files.

## **Dependencies**
Before running the script, ensure the following dependencies are installed:

- `torch`
- `torchvision`
- `scipy`
- `numpy`
- `tqdm`
- `argparse`

You can install the required dependencies using `pip`:
```bash
pip install torch torchvision scipy numpy tqdm
```

## **Usage**

1. **Prepare the dataset**:
   Ensure that your dataset (e.g., ShapeNetCore) is in the correct format and located at the path specified in the `--dataset_path` argument. The dataset should be in `.h5` format for use with the `ShapeNetCore` class.

2. **Download the pretrained models**:
   - Download the pretrained FlowVAE model and save it to the path specified in the `--ckpt` argument (default is `./pretrained/GEN_chair.pt`).
   - The PointNet model (`best_model.pth`) should also be downloaded and placed in the working directory.

3. **Set the desired arguments**:
   The script uses argparse to specify various parameters. You can modify them in the command line or adjust the default values in the script.

   Example:
   ```bash
   python script.py --ckpt ./pretrained/GEN_chair.pt --dataset_path ./data/modelnet.h5 --batch_size 1 --sample_num_points 2048 --normalize shape_bbox
   ```

   **Arguments**:
   - `--ckpt`: Path to the pretrained FlowVAE model (e.g., `./pretrained/GEN_chair.pt`).
   - `--categories`: List of categories to sample from in the dataset (e.g., `['desk1']`).
   - `--save_dir`: Directory to save the output files (e.g., `./results`).
   - `--device`: Device to run the model on (e.g., `cuda` or `cpu`).
   - `--latent_dim`: Latent dimension for the VAE model (default is `256`).
   - `--dataset_path`: Path to the dataset (e.g., `./data/modelnet.h5`).
   - `--batch_size`: Batch size for data loading (default is `1`).
   - `--scale_mode`: Mode for scaling the point clouds (e.g., `shape_unit` or `shape_bbox`).
   - `--sample_num_points`: Number of points to sample from the point cloud (default is `2048`).
   - `--normalize`: Mode for normalization of point clouds (e.g., `shape_unit`, `shape_bbox`, or `None`).
   - `--seed`: Random seed for reproducibility (default is `500`).

4. **Run the script**:
   After setting the arguments, you can run the script. The script will:
   - Load the dataset and preprocess the point clouds.
   - Generate adversarial point clouds using the FlowVAE model and the PointNet classification model.
   - Calculate Chamfer Distance, Hausdorff Distance, and MSE between the generated and reference point clouds.
   - Save the generated and reference point clouds as `.npy` files.
   - Log the performance metrics (MSE, CDC, CD, HD) to a log file.

   Example:
   ```bash
   python generate_adversarial_pointclouds.py --ckpt ./pretrained/GEN_chair.pt --categories desk1 --save_dir ./results --dataset_path ./data/modelnet.h5 --batch_size 1
   ```

5. **Output**:
   - **Generated Point Clouds**: The generated point clouds will be saved in the specified `save_dir` as `out.npy`.
   - **Reference Point Clouds**: The reference point clouds (from the dataset) will be saved as `ref.npy`.
   - **Log File**: Performance metrics such as MSE, CDC, CD, and HD will be logged in a log file in the `save_dir`.

   Example log output:
   ```
   MSE: 0.123456789012
   CDC: 0.987654321098
   CD: 0.234567890123
   HD: 0.345678901234
   class: x: 3, ref: 5
   ```

6. **Adversarial Attack**:
   The script also includes functionality to generate adversarial samples. It optimizes the noise added to the generated point cloud in an attempt to mislead the classification model.

   If the attack is successful (i.e., the predicted class of the generated point cloud differs from the reference class), the script will output a message such as:
   ```
   Attack successful: prediction1.max(dim=1)[1] != prediction.max(dim=1)[1]
   ```

---

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

## **Acknowledgments**
- This code uses the PointNet and FlowVAE models for point cloud generation and classification.
- Special thanks to the authors of the datasets and models used in this project.
