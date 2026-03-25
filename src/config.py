class Config:
    batch_size = 4              #placeholder    
    learning_rate_student = 1e-4    
    learning_rate_seg = 1e-4        
    weight_decay = 1e-5
    num_epochs_student = 50
    num_epochs_seg = 50

    # Model settings
    input_size = (256, 256)         
    encoder_name = "resnet18"
    pretrained_teacher = True
    freeze_teacher = True
    freeze_student_seg = True        
    
    # Synthetic anomaly generation
    perlin_threshold = 0.5          
    beta_range = (0.15, 1.0)         # blending factor

    # Loss settings
    focal_gamma = 2.0                
    seg_downsample_ratio = 0.25      
