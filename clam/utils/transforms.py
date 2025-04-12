import torchvision.transforms as transforms

def get_transform(is_training=True):
    """Get the transformation pipeline for CLAM model
    
    Args:
        is_training: Whether to use training transforms (with augmentation)
        
    Returns:
        A Compose object with the transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_basic_transform():
    """Get a basic transform without augmentation
    
    Returns:
        A Compose object with the basic transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 