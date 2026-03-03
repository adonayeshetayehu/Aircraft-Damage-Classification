def train_model(model, train_gen, valid_gen, n_epochs):
    history = model.fit(train_gen, epochs=n_epochs, validation_data=valid_gen)
    return history.history