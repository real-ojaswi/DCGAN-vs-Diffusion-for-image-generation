This repository consists the code for the implementation of DCGAN and its comparison with Diffusion for face image generation. Both DCGAN and Diffusion have been trained on a subset of [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). The code for diffusion inside the GenerativeDiffusion is borrowed from [DLStudio](https://engineering.purdue.edu/kak/distDLS/DLStudio-2.4.2.html) as also mentioned in the file itself. 

The images generated from GAN and Diffusion are compared quantitatively using FID Score. The FID Score was obtained to be 80.96 for images generated using DCGAN and 48.96 for images generated using Diffusion. The images generated are shown below. The direct comparison might not be fair as the training and hyperparameter tuning might not have been enough (especially for DCGAN) given computational and time constraints.

By DCGAN:
![generated_image_5](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/6b4b5073-b0cc-4448-968e-66ecb22482c2)
![generated_image_6](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/adcad883-a2e1-4398-8874-3aece4711299)
![generated_image_7](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/9e15079b-3def-4232-9956-682c79587ea2)
![generated_image_8](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/0e4f5bd5-b8bb-41a3-a6ef-a9dbb80138e0)


By Diffusion:
![test_2](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/ca9f1e6b-983f-4aa6-a081-fed02a7d7dfb)
![test_0](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/2c7ddd12-2d5e-403f-b90d-d1f0fdb450e3)
![test_1](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/c730a9ec-1b68-4954-aed1-184d1dccc26f)
![test_4](https://github.com/thenoobcoderr/DCGAN-vs-Diffusion-for-face-image-generation/assets/139956609/d2d4fac6-c4a2-4e84-befa-66596dd454bd)
