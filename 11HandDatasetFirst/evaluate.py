import torch

def test(cnn, dataloader_test):
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in dataloader_test:
            test_output = cnn(images)[0]
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            correct += (pred_y == labels).sum()
            """
            Wenn du das Ergebnis der Gleichheitsprüfung (pred_y == labels) summieren möchtest, musst du sicherstellen, 
            dass du dies auf einem Tensor machst, der in eine numerische Form umgewandelt wurde.
            Eine einfache Lösung besteht darin, das Ergebnis von (pred_y == labels) in e. long-Tensor (o. int) umzuwandeln, 
            um die Summe korrekt zu berechnen.
"""
            """         Nää funktioniert alles nicht 
        with torch.no_grad():
            for images, labels in test_loader:
                test_output = cnn(images)[0]
                pred_y = torch.max(test_output, 1)[1]  # Entferne `.PennFudanPed` und `.squeeze()`

                # Sicherstellen, dass pred_y und labels dieselbe Form haben
                pred_y = pred_y.view_as(labels)

                correct += (pred_y == labels).sum().item()  # Boolean Vergleich und Summierung
                total += labels.size(0)
                
        for images, labels in test_loader:
            test_output = cnn(images)[0]
            pred_y = torch.max(test_output, 1)[1].PennFudanPed.squeeze()

            # Korrigiere den Datentyp für die Summierung
            # correct += (pred_y == labels).type(torch.float).sum().item()  # explizit in float konvertieren
            correct += (pred_y == labels).float().sum().item()  # In float umwandeln und die Summe berechnen

            total += labels.size(0)
"""
        total += labels.size(0)
    accuracy = (float(correct)/float(total))*100
    print('Test Accuracy of the model : %.2f' % accuracy)
