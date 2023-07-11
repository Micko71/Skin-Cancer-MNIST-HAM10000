
## Skin-Cancer-MNIST-HAM10000
## Transfer learning  is applied to solve an image classifiction problem
### Data Preparation
The metadata was split 60:40 into training and validation/test sets and the os.mkdir() method was used to create subfolders for each image class within the training and validatioin/test folders. The shutil.copyfile() method was then used to populate each subfolder with its corresponding images as described by the metadata. Examination of the metadata showed the classes to be highly imbalanced.

All images were resized and normalized to be compatible with the inputs expected by the pytorch pretrained models. Random rotation and random horizontal flip were applied to the training data.  The images were loaded using the pytorch imagefolder class. To mitigate the effects of the class imbalance the WeightedRandomSampler method was applied and added as a parameter in the dataloader for the training set. This essentially uses resampling to ensure that the model sees an approximately equal representation of observations from each class in each training batch. 

To identify the best model during training it was also necessary to consider the effect of the class imbalance. For this reason, the minimum validation loss (rather than overall accuracy) was used as the criterion for selecting the final model.  The weighted loss function was applied to the validation set to account for effect of the class imbalance on the validation loss.

The validation/test data was split 50:50 into validation and test sets using the torch.utils.data random_split method and passed to a pytorch dataloader for iteration through the model in batches. Weights for the validation loss function were calculated based on the class distribution of the test data.

The MobileNet_V3 model was loaded from the torchvision.modules subpackage along with its pretrained weights  to allow for transfer
learning. The model parameters were not frozen meaning all parameters could be updated during training. The output of the final linear layer of each model was modified to be compatible with the number of classes in the dataset. Cross entropy loss was
chosen as the loss function along the Adam optimizer and its default learning rate of 0.001.

The best model was identified at the end of epoch 6.  The Model achieved an overall accuracy of 75% and a macro averaged f1-score of 0.66 with individual class accuracies varying from 54% to 96%.
