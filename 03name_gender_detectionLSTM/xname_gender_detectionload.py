import torch
from name_gender_detectionlstm import load_checkpoint
from tqdm import tqdm
from matplotlib import pyplot as plt

plt.title("LOSS Curve")

plt.plot(plot_loss_train, label='Train Loss')
plt.plot(plot_loss_valid, label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
plt.savefig('/home/boris.grillborzer/PycharmProjects/00CNNundRNN/boris.grillborzer/03name_gender_detectionLSTM/gender_detection_loss_curve.png')

# from google.colab import files
#
# files.download('/content/drive/MyDrive/gender_detection_loss_curve.png')

name = 'boris'
name_tensor = torch.tensor([indexing(name, CHAR2INDEX)], dtype=torch.int).to(DEVICE)
print(name_tensor.size())
prediction = MODEL(name_tensor)
print(prediction)
prediction = torch.round(prediction).item()
print(prediction)
print(f'The Gender is => {INDEX2LABEL[prediction]}')

"""# **Test Set**"""

MODEL.eval()

with torch.no_grad():
    test_true_labels = []
    test_predicted_labels = []

    for batch in DATALOADER_TEST:
        DATA_BATCH, LABEL_BATCH = batch

        DATA_BATCH = DATA_BATCH.to(DEVICE)
        LABEL_BATCH = LABEL_BATCH.to(DEVICE)

        out = MODEL(DATA_BATCH)

        test_true_labels.extend(LABEL_BATCH.tolist())
        test_predicted_labels.extend(torch.round(out).tolist())

    ACCURACY = binary_acc(y_pred=test_predicted_labels, y_test=test_true_labels)

    print(f'Accuracy on Test Set is {ACCURACY :.2f}%')