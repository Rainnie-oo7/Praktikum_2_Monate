from char_rnn_generation_tutorial import *

# Initialisiere das RNN wie zuvor
rnn = RNN(n_letters, 128, n_letters)

# Lade das Modell, falls es schon trainiert wurde
rnn.load_state_dict(torch.load('04best_model.pth'))

# Setze das Modell in den Evaluierungsmodus (deaktiviert Dropout, BatchNorm etc.)
rnn.eval()


samples('Russian', 'RUS')

samples('German', 'GER')

samples('Spanish', 'SPA')

samples('Chinese', 'CHI')