import numpy as np
import torch

from src import utils


class FF_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_CIFAR10_partition(opt, partition)
        self.num_classes = num_classes
        # Scale label to provide strong signal relative to normalized pixels
        # With 3072 pixels vs 10 label features, need strong scaling for label signal
        self.label_scale = 75.0
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes * self.label_scale

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        """Concatenate one-hot label to flattened image for positive sample."""
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).float() * self.label_scale  # Scale label to match pixel magnitude
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Concatenate label at the beginning: [num_classes + C*H*W]
        pos_sample = torch.cat([one_hot_label, flat_sample], dim=0)
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        """Concatenate wrong one-hot label to flattened image for negative sample."""
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).float() * self.label_scale  # Scale label to match pixel magnitude
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Concatenate wrong label at the beginning: [num_classes + C*H*W]
        neg_sample = torch.cat([one_hot_label, flat_sample], dim=0)
        return neg_sample

    def _get_neutral_sample(self, sample):
        """Concatenate uniform label distribution to flattened image for neutral sample."""
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Concatenate uniform label at the beginning: [num_classes + C*H*W]
        neutral_sample = torch.cat([self.uniform_label, flat_sample], dim=0)
        return neutral_sample
    
    def _get_all_sample(self, sample):
        """Create samples with all possible class labels concatenated."""
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Create tensor to hold all class variations: [num_classes, num_classes + C*H*W]
        all_samples = torch.zeros((self.num_classes, self.num_classes + flat_sample.shape[0]))
        for i in range(self.num_classes):
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(i), num_classes=self.num_classes
            ).float() * self.label_scale  # Scale label to match pixel magnitude
            # Concatenate each possible label with the image
            all_samples[i] = torch.cat([one_hot_label, flat_sample], dim=0)
        return all_samples

    def _generate_sample(self, index):
        # Get CIFAR10 sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label

class FF_MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.mnist = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z
    
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()
        return all_samples

    def _generate_sample(self, index):
        # Get MNIST sample.
        sample, class_label = self.mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label


class FF_FashionMNIST(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.fashion_mnist = utils.get_FashionMNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.fashion_mnist)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )
        neg_sample = sample.clone()
        neg_sample[0, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z
    
    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes, sample.shape[0], sample.shape[1], sample.shape[2]))
        for i in range(self.num_classes):
            all_samples[i, :, :, :] = sample.clone()
            one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(i), num_classes=self.num_classes)
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label.clone()
        return all_samples

    def _generate_sample(self, index):
        # Get FashionMNIST sample.
        sample, class_label = self.fashion_mnist[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label


class FF_STL10(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.stl10 = utils.get_STL10_partition(opt, partition)
        self.num_classes = num_classes
        # Scale label to provide strong signal relative to normalized pixels
        # With 3072 pixels (32x32x3 downsampled, same as CIFAR-10), use same scaling as CIFAR-10
        self.label_scale = 75.0
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes * self.label_scale

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, all_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.stl10)

    def _get_pos_sample(self, sample, class_label):
        """Concatenate one-hot label to flattened image for positive sample."""
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).float() * self.label_scale  # Scale label to match pixel magnitude
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Concatenate label at the beginning: [num_classes + C*H*W]
        pos_sample = torch.cat([one_hot_label, flat_sample], dim=0)
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        """Concatenate wrong one-hot label to flattened image for negative sample."""
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).float() * self.label_scale  # Scale label to match pixel magnitude
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Concatenate wrong label at the beginning: [num_classes + C*H*W]
        neg_sample = torch.cat([one_hot_label, flat_sample], dim=0)
        return neg_sample

    def _get_neutral_sample(self, sample):
        """Concatenate uniform label distribution to flattened image for neutral sample."""
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Concatenate uniform label at the beginning: [num_classes + C*H*W]
        neutral_sample = torch.cat([self.uniform_label, flat_sample], dim=0)
        return neutral_sample
    
    def _get_all_sample(self, sample):
        """Create samples with all possible class labels concatenated."""
        # Flatten image: [C, H, W] -> [C*H*W]
        flat_sample = sample.reshape(-1)
        # Create tensor to hold all class variations: [num_classes, num_classes + C*H*W]
        all_samples = torch.zeros((self.num_classes, self.num_classes + flat_sample.shape[0]))
        for i in range(self.num_classes):
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(i), num_classes=self.num_classes
            ).float() * self.label_scale  # Scale label to match pixel magnitude
            # Concatenate each possible label with the image
            all_samples[i] = torch.cat([one_hot_label, flat_sample], dim=0)
        return all_samples

    def _generate_sample(self, index):
        # Get STL10 sample.
        sample, class_label = self.stl10[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        all_sample = self._get_all_sample(sample)
        return pos_sample, neg_sample, neutral_sample, all_sample, class_label