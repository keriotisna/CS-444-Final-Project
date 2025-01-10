# CS 444 Final Project
--- 

Benchmarking project which examines a variety of different model architecture and data augmentation performances on the CIFAR-10 dataset.

Data augmentation is a crucial part of reliable models, as the bias variance noise tradeoff states that as model complexity increases, so does the variance of its predictions which is often harmful to predictive accuracy. While machine learning models have gotten more powerful over time, their increased complexity is not without drawbacks, as models often exhibit even greater variance. To combat this drawback of larger models, we can utilize a technique called data augmentation which serves to synthetically expand the dataset by adding artificial variance. The larger augmented dataset is effective at reducing model variance by forcing the model to generalize which helps it identify the common features we actually want it to focus on instead of overfitting to the training set.

![ML--Bias-Vs-Variance-(1)](https://github.com/user-attachments/assets/b916031f-258d-4b38-99b9-81dcc8afb1d0)

- A diagram showing the bias-variance tradeoff. As model complexity increases, there is a similar increase in model error due to variance.

![1_ae1tW5ngf1zhPRyh7aaM1Q](https://github.com/user-attachments/assets/ba80dc16-872f-434f-8419-40558dd593fb)

- Examples of sample data augmentations that can be applied to image datasets.

![easyaugment(1)](https://github.com/user-attachments/assets/303d0d21-3a40-473a-8c09-b435f1af18b5)

- Samples from the CIFAR-10 dataset that have undergone a variety of augmentations like rotations, color jitter, and scaling.

![image](https://github.com/user-attachments/assets/44b7b1b0-b298-4dd3-b799-0b0cf601c14f)

- A table showing how baseline model performance is affected by augmentation. While more complex augmentations can harm performance, they are also able to prevent the model from overfitting and push the validation accuracy above the training accuracy when training time and augmentation complexity are increased.
