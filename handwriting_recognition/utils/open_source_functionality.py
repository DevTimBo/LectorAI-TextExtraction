# Author Tim Harmling & Jason Pranata
import torch
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def load_os_dataset(train_dir, val_dir, processor):
    train_df_list = os.listdir(train_dir)
    val_df_list = os.listdir(val_dir)

    train_df_jpg_list = [train_df_list[i] for i in range(len(train_df_list)) if train_df_list[i].endswith('.jpg')]
    val_df_jpg_list = [val_df_list[i] for i in range(len(val_df_list)) if val_df_list[i].endswith('.jpg')]

    train_df = pd.DataFrame(columns=['file_name', 'text'])
    val_df = pd.DataFrame(columns=['file_name', 'text'])

    for i in range(len(train_df_jpg_list)):
        text_file = f"{train_df_jpg_list[i].split('.')[0]}.txt"
        with open(os.path.join(train_dir, text_file), 'r') as f:
            text = f.read()
        train_df.loc[i] = {'file_name': train_df_jpg_list[i], 'text': text.replace('|', ' ')}

    for i in range(len(val_df_jpg_list)):
        text_file = f"{val_df_jpg_list[i].split('.')[0]}.txt"
        with open(os.path.join(val_dir, text_file), 'r') as f:
            text = f.read()
        val_df.loc[i] = {'file_name': val_df_jpg_list[i], 'text': text.replace('|', ' ')}

    train_df.head()
    val_df.head()
    return create_dataset(processor, train_dir, train_df, val_dir, val_df)

def create_dataset(processor, train_dir, train_df, val_dir, val_df, augment=False, show_image=True):
    train_dataset = Dataset(root_dir=train_dir,
                            df=train_df,
                            processor=processor, augment=augment)
    eval_dataset = Dataset(root_dir=val_dir,
                            df=val_df,
                            processor=processor)

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    # Plot Example Image
    if show_image:
        plt.imshow(train_dataset[0]["pixel_values"].permute(1, 2, 0))
        print("Image shape:", train_dataset[0]["pixel_values"].shape)
        plt.axis("off")
        
    return train_dataset, eval_dataset

class Dataset:
    def __init__(self, root_dir, df, processor, max_target_length=128, augment=False, target_size=(1024, 128)):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.augment = augment
        self.target_size = target_size


        if augment:
            self.augment_transforms = transforms.Compose([
                transforms.RandomRotation(2),  
                transforms.ColorJitter(brightness=0.25, contrast=0.25),  # Randomly change brightness, contrast, etc.
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df['file_name'][idx]
        text = str(self.df['text'][idx])

        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")

        if self.augment:
            image = self.augment_transforms(image)


        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length, truncation=True).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}

def plot_history(trainer, save_model_name, show_plot=False):
    # Assuming you have completed training and your trainer object is named 'trainer'
    # Extracting metrics from the log history
    log_history = trainer.state.log_history
    epochs = []
    train_epochs = []
    levenshtein_distances = []
    train_losses = []
    eval_losses = []
    for entry in log_history:
        if 'eval_loss' in entry: 
            epochs.append(entry['epoch'])
            levenshtein_distances.append(entry['eval_levenshtein'])
        
            eval_losses.append(entry["eval_loss"])
        if 'loss' in entry:
            train_losses.append(entry["loss"])
            train_epochs.append(entry['epoch'])

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, levenshtein_distances, marker='.', linestyle='-', color='g')
    plt.plot(train_epochs, train_losses, marker='.', linestyle='-', color='r')
    plt.plot(epochs, eval_losses, marker='.', linestyle='-', color='b')
    plt.legend(["Levenshtein Distance", "Training Loss", "Eval Loss"])
    plt.title(f'Model: {os.path.basename(save_model_name)} Levenshtein: {round(min(levenshtein_distances),2)} Loss: {round(min(train_losses),2)} Eval Loss: {round(min(eval_losses),2)} ')
    plt.xlabel('Training Epochs')
    plt.ylabel(["Levenshtein Distance", "Training Loss", "Eval Loss"])
    plt.grid(True)
    plt.savefig(f"{save_model_name}/results.png")
    if show_plot:
        plt.show()

def try_model(model, processor, val_df, eval_dataset):
    print("Predicted; True")
    for i, eval in enumerate(eval_dataset):
        pixel_values = eval['pixel_values'].unsqueeze(0)
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        real_text = val_df['text'][i]
        print(generated_text, real_text)
        