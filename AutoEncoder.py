import torch
from torch import nn, set_default_device, Generator, save
from torch.cuda import is_available
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


def visualize_images(input_images: torch.Tensor, output_images: torch.Tensor, labels, num_samples=4):
    input_images = input_images.cpu().numpy()
    output_images = output_images.cpu().numpy()
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axes[0, i].imshow(input_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Input {labels[i]}')

        axes[1, i].imshow(output_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Output')
    plt.show()


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.test_loader: DataLoader = None
        self.train_loader: DataLoader = None
        self.test_dataset = None
        self.train_dataset = None

        self.DEVICE = 'cuda' if is_available() else 'cpu'
        set_default_device(self.DEVICE)

        self.encoder = nn.Sequential(
            # input [Batch_N,1,28,18]
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  #[32,14,14]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  #[64,7,7]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5)  # [128,3,3]

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 5),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def load_model(self, PATH):
        super().load_state_dict(torch.load(PATH))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def load_MNIST_model(self, train_batch=128, test_batch=32):
        self.train_dataset = datasets.MNIST(root="~/torch_datasets",
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)
        self.test_dataset = datasets.MNIST(root="~/torch_datasets",
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=train_batch,
                                       generator=Generator(self.DEVICE),
                                       shuffle=True,
                                       num_workers=0)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=test_batch,
                                      generator=Generator(self.DEVICE),
                                      shuffle=False,
                                      num_workers=0)

    def start_train(self, n_epoch, optimizer, criterion, dataset_type='train', visualize_process=False,
                    model_save_PATH="",
                    save_model_each_epoch_step=4):
        if self.test_loader is None or self.train_loader is None:
            raise ValueError("You not provided a train/test loader!")

        else:
            if dataset_type == 'train':
                target_dataset = self.train_loader
            else:
                target_dataset = self.test_loader

            super().train()
            Losses = []
            for epoch in range(n_epoch):
                loss = 0
                for img_batch, label_label in target_dataset:
                    if img_batch.shape[0] == 128:
                        img_in = img_batch.to(self.DEVICE)
                        img_output = self.forward(img_in)
                        optimizer.zero_grad()
                        train_loss = criterion(img_in, img_output)
                        train_loss.backward()
                        optimizer.step()
                        loss += train_loss.item()
                        Losses.append(loss / len(target_dataset))
                print(f'epoch : {epoch + 1}/{n_epoch}, loss = {Losses[-1]:.16f}')

                if (epoch + 1) == n_epoch:
                    save(super().state_dict(), model_save_PATH + 'model_epoch_' + str(epoch + 1) + '.pth')
                    print(f'saving model at epoch  {epoch + 1}')

                if (epoch + 1) % save_model_each_epoch_step == 0 and (epoch + 1) != n_epoch:
                    save(super().state_dict(), model_save_PATH + 'model_epoch_' + str(epoch + 1) + '.pth')
                    print(f'saving model at epoch  {epoch + 1}')

    def visualize(self, samples):
        super().eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.DEVICE)
                reconstructed_images = self.forward(images)
                visualize_images(images, reconstructed_images, num_samples=samples, labels=labels)
                break