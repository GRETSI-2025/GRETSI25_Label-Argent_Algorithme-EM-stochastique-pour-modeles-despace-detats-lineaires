import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from em import run_em_algorithm
from utils import load_data, save_results, plot_results, forward_operator, save_image

def main():
    print("Loading data...")
    # Load data from the .npz file
    data = load_data('./data/data.npz')
    if data is None:
        print("Data loading failed.")
        return

    # Extract arrays from the data
    model_images = data['model_images']    # Ground truth image(s)
    visibilities = data['visibilities']
    uvw = data['uvw']
    dirty = data['dirty']                  # Dirty image(s)
    freq = data['freq']
    tau = data['texture_value']
    npixel = 64  # assuming image dimensions are 64x64

    # Compute the forward operator
    FOp = forward_operator(uvw, freq, npixel, cellsize=None)
    print("Starting EM algorithm...")

    # Run the EM algorithm for 2 iterations (adjust as needed)
    model, smoothed_estimates, MMSE = run_em_algorithm(visibilities, dirty, FOp, model_images, tau, n_iterations=1)

    # Compute quality metrics comparing the final smoothed estimate with the ground truth.
    # Here model_images is the ground truth, and smoothed_estimates[-1] is the estimated image.
    ssim_mmse = ssim(np.real(smoothed_estimates[-1]), model_images, data_range=model_images.max() - model_images.min())
    psnr_mmse = psnr(np.real(smoothed_estimates[-1]), model_images, data_range=model_images.max() - model_images.min())
    print(f"SSIM (MMSE): {ssim_mmse}, PSNR (MMSE): {psnr_mmse}")

    # Save results (e.g., model parameters) using the utility function
    results = model  # or any other formatted results
    save_results(results, output_dir='results')
    
    # Optionally, plot results via the utility function
    # plot_results(dirty)

        
    save_image(model_images[-1], 'results/model_image.jpg')
    save_image(dirty, 'results/dirty_image.jpg')
    save_image(np.real(smoothed_estimates[-1][-1]), 'results/estimated_image.jpg')


    # Show figures interactively
    plt.show()

if __name__ == '__main__':
    main()
