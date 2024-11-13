import torch
from names2 import *
from data import *
import sys

# rnn = torch.load('char-rnn-classification.pt')

# Instanziieren Sie das Modell
rnn = RNN(n_letters, n_hidden, n_categories)

# Laden Sie die gespeicherten Gewichte
rnn.load_state_dict(torch.load('model.pt'))

# Setzen Sie das Modell in den Evaluierungsmodus (deaktiviert Dropout/BatchNorm)
rnn.eval()

# # Just return an output given a line
# def evaluate(line_tensor):
#     hidden = rnn.initHidden()
#
#     for i in range(line_tensor.size()[0]):
#         output, hidden = rnn(line_tensor[i], hidden)
#
#     return output
#3. Verwenden des Modells für Vorhersagen
#
# Nach dem Laden des Modells können Sie es direkt für Vorhersagen verwenden, ohne erneut zu trainieren. Hier ein Beispiel für die Vorhersage:
#
# # Beispiel für eine Eingabe
# input_tensor = lineToTensor('Albert')  # lineToTensor konvertiert einen String in einen Tensor
# hidden = rnn.initHidden()  # Initialisieren Sie den Hidden State
#
# # Durchlaufen Sie die Eingabezeichen
# for i in range(input_tensor.size()[0]):
#     output, hidden = rnn(input_tensor[i], hidden)
#
# # Ausgabe anzeigen
# print(output)


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')

#
# if __name__ == '__main__':
#     predict('Dovesky')
#     predict('Jackson')
#     predict('Satoshi')
    #predict(sys.argv[1])