from scipy.io import loadmat, savemat

# Lade die .mat-Datei
mat_file_path = '/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/data/CARDS_COURTYARD_B_T/polygons.mat'
data = loadmat(mat_file_path)

# Die Arrays, die du ändern möchtest, in einer Liste speichern
arrays_to_modify = ['myleft', 'myright', 'yourleft', 'yourright']

for i in range(99):  # Bis 100 Iterationen
    for array_name in arrays_to_modify:
        # Setze den Wert des Arrays an Position [0, i] auf z. B. eine gewünschte Zahl oder 1
        try:
            out = data[array_name][0, i]
            print("out", out)
            print("PennFudanPed[array_name][0, i]", data[array_name][0, i])
        except IndexError:
            print(f"{array_name} hat keine ausreichende Größe für Index [0, {i}].")

# Speichere die aktualisierte .mat-Datei
updated_mat_file_path = 'updated_file.mat'
savemat(updated_mat_file_path, data)

print(f"Die Werte wurden erfolgreich aktualisiert und in '{updated_mat_file_path}' gespeichert.")
