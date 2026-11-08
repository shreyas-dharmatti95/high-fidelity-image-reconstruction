# High-Fidelity Image Reconstruction

Image reconstruction with **Random Fourier Features (RFF) + linear regression** and **low-rank matrix factorization**. An image is treated as a function from pixel coordinates to colour, and models learn that mapping to reconstruct, up-sample, fill in and compress images. Quality is measured with **RMSE/MSE and PSNR**.

Built as part of the ES 335 (Machine Learning) course at IIT Gandhinagar, on top of the course's reference notebooks (see Credits).

---

## Approach

**Coordinates → colour with RFF.** Each pixel's `(x, y)` coordinate is scaled to `[-1, 1]` and lifted into a high-dimensional space with random Fourier features (scikit-learn's `RBFSampler`). A single linear layer then predicts the `(R, G, B)` value, trained with Adam on MSE loss. Plain linear regression on raw coordinates cannot represent image detail; RFF makes it expressive.

**Low-rank matrix factorization.** A channel is approximated as `W · H` with a small rank `r`, learned by gradient descent with a loss that ignores missing (NaN) pixels. This is used both to fill in missing pixels and to compress image patches.

## Notebooks and results

| Folder | Task | Key setup | Result |
|---|---|---|---|
| [`task2_rff_image_reconstruction`](task2_rff_image_reconstruction) | Reconstruct a 300×300 crop from its coordinates | 20,000 RFF features, σ = 0.008, 2,500 epochs | RMSE **0.179** with plain linear regression, **0.043** with RFF |
| [`task3_rff_super_resolution`](task3_rff_super_resolution) | 2× super-resolution: learn from a 200×200 image, predict a 400×400 one | 50,000 RFF features, σ = 0.008, 1,500 epochs, lr 0.005 | RMSE **0.0295**, PSNR **30.59 dB**, 91.4% of pixels within ±0.05 |
| [`task4_matrix_factorization_missing_pixels`](task4_matrix_factorization_missing_pixels) | Reconstruct an image with a missing 30×30 patch (`Task_4_patch`) or 900 randomly missing pixels (`Task_4_random`) using matrix factorization, then compare with RFF + linear regression | Rank sweeps; in the random-pixel notebook RMSE levels off around rank 90, so ranks 80 and 90 are used | RMSE/PSNR per rank are shown in the notebook plots |
| [`task5_matrix_factorization_compression`](task5_matrix_factorization_compression) | Compress 50×50 image patches into two thin matrices | Ranks 5, 10, 25 and 50 | Reconstructed patches with RMSE/PSNR per rank in the notebook |

RMSE and PSNR are computed on pixel values scaled to `[0, 1]`, with PSNR = 20·log10(1/RMSE). The matrix-factorization notebooks fit each colour channel separately with Adam (learning rate 0.01, 1,000 iterations).

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

Each notebook reads its image from its own folder (`image.jpg` or `dog.jpg`), so open a notebook from inside its folder. Device selection is in each notebook's setup cell (CPU by default, with CUDA or Apple MPS where the notebook supports it).

## Images

The sample images are included only so the notebooks run as they are. `dog.jpg` is the corgi photo from the Segment Anything demo gallery (originally an Adobe Stock image). If you reuse this repository, swap in your own images.

## Credits

- Built on the course's reference notebooks by Nipun Batra: [RFF / SIREN image reconstruction](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/siren.ipynb) and [matrix factorization](https://github.com/nipunbatra/ml-teaching/blob/master/notebooks/movie-recommendation-knn-mf.ipynb).
- Random Fourier Features: Rahimi & Recht, *Random Features for Large-Scale Kernel Machines* (2007).
