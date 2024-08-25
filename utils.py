from pytorch_fid.fid_score import calculate_activation_statistics, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import os
import cv2
import torch
import numpy

class calculateFID():
    def __init__(self, device='cpu'):
        self.device=device
        
    def __call__(self, real_path, fake_path,):
        real_images= os.listdir(real_path)
        fake_images= os.listdir(fake_path)
        real_images_full= [os.path.join(real_path, image_path) for image_path in real_images]
        fake_images_full= [os.path.join(fake_path, image_path) for image_path in fake_images]
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(self.device)
        m1, s1 = calculate_activation_statistics(real_images_full, model, device=self.device)
        m2, s2 = calculate_activation_statistics(fake_images_full, model, device=self.device)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print(f'FID: {fid_value:.2f}')
        return fid_value

class image_generator():
    def __init__(self, generator, generator_weight= None, device='cpu') -> None:
        self.generator= generator
        self.generator_weight= generator_weight
        self.device= device

        if generator_weight is not None:
            self.load_model_weights(self.generator, self.generator_weight)
        self.generator.eval()

    def load_model_weights(self, model, checkpoint_file):
        """
        Load model weights from a checkpoint file if it exists.
        """
        if os.path.isfile(checkpoint_file):
            print(f'Weight file found for {model.__class__.__name__}. Using the saved weights!')
            model.load_state_dict(torch.load(checkpoint_file))
        else:
            raise ValueError(f'Weight file not found for {model.__class__.__name__}')
        
    def generate(self, num_images, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        noise = torch.randn(num_images, 100, 1, 1, device=self.device)    
        with torch.no_grad():
            fakes = self.generator(noise).detach().cpu()
            # grid= numpy.transpose(torchvision.utils.make_grid(fakes, padding=1, pad_value=1, normalize=True).cpu(), (1,2,0))
            # plt.imshow(grid)
    

        for i, fake in enumerate(fakes):
            # Convert the tensor to a PIL image
            
            fake_np = fake.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC format
            fake_np_rescaled= (fake_np+1)/2
            
            # Convert the numpy array to the appropriate data type (e.g., uint8)
            fake_np_uint8 = (fake_np_rescaled * 255).astype(numpy.uint8)  # Scale to [0, 255] range


            # Convert the numpy array to BGR format
            fake_bgr = cv2.cvtColor(fake_np_uint8, cv2.COLOR_RGB2BGR)

            # Save the image using OpenCV
            image_path = os.path.join(save_dir, f'generated_image_{i+1}.png')
            cv2.imwrite(image_path, fake_bgr)

            print(f'Saved generated image {i+1} at {image_path}')
        
            



        