def train(n_epochs, model, data_train, accumulation_steps=4):
    n_batches = len(data_train)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(n_epochs):
        for batch, (images, labels) in enumerate(data_train):
            # Vorwärtsdurchlauf
            output = model(images)[0]
            loss_dict = model(images, labels)  # loss_dict kommt von deinem Modell
            losses = sum(loss for loss in loss_dict.values())  # Summe der Verluste

            # Verlust für Gradient Accumulation aufteilen
            loss = losses / accumulation_steps

            # Backpropagation
            loss.backward()

            # Akkumulationsschritte überprüfen
            if (batch + 1) % accumulation_steps == 0:
                optimizer.step()  # Optimierungsschritt
                optimizer.zero_grad()  # Gradienten zurücksetzen

            # Ausgabe der Zwischenschritte
            if (batch + 1) % 100 == 0:
                print(f"Epoch [{epoch + 1}/{n_epochs}], Step [{batch + 1}/{n_batches}], Loss: {losses.item():.4f}")

        # Am Ende jeder Epoche Gradienten zurücksetzen (optional, zur Sicherheit)
        optimizer.zero_grad()
