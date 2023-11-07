import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# Define your GAN model architecture as a Python class
class GANModel(nn.Module):
    def __init__(self):
        super(GANModel, self).__init()
        # Replace this with your actual GAN model architecture
        self.generator = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        generated_image = self.generator(x)
        return generated_image

# Initialize the Tkinter window
window = tk.Tk()
window.title("Image Restoration App")

# Create a function to open the image file dialog
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the selected image
        image = Image.open(file_path)
        # Display the selected image in a Tkinter label
        image = ImageTk.PhotoImage(image)
        image_label.config(image=image)
        image_label.image = image

# Create a function to restore the image using your GAN model
def restore_image():
    if image_label.image:
        input_image = Image.open(filedialog.askopenfilename())
        input_image = transforms.ToTensor()(input_image)
        input_image = input_image.unsqueeze(0)

        # Load the GAN model and pre-trained weights
        gan_model = GANModel()
        checkpoint = torch.load("GFPGANv1.pth")  # Replace with the actual path to your model checkpoint file
        gan_model.load_state_dict(checkpoint['state_dict'])
        gan_model.eval()

        # Perform image restoration
        with torch.no_grad():
            restored_image = gan_model(input_image)

        # Post-process the restored image
        restored_image = restored_image.squeeze(0).clamp(0, 1).numpy()
        restored_image = (restored_image * 255).astype('uint8')

        # Display the restored image
        restored_image = Image.fromarray(restored_image)
        restored_image_label.config(image=ImageTk.PhotoImage(restored_image))
        restored_image_label.image = ImageTk.PhotoImage(restored_image)

# Create a button to open the input image file dialog
open_button = tk.Button(window, text="Open Input Image", command=open_image)
open_button.pack()

# Create a button to restore the image
restore_button = tk.Button(window, text="Restore Image", command=restore_image)
restore_button.pack()

# Create labels to display the input and restored images
image_label = tk.Label(window)
image_label.pack()
restored_image_label = tk.Label(window)

# Start the Tkinter main loop
window.mainloop()
