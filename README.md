# Training a neural network able to recognize Facial Expressions with Keras

Description available in English and Italian below.

## English

Code to train a neural network capable of recognizing facial expressions with Keras, totally commented in English. It was written for my Master's Degree thesis. It must be executed in this order:

1. **CommonVariables.py**: contains common variables between the different files. It must be modified in order to specify the train and test folders.
2. **Preprocessing.py**: allows you to preprocess input images. Some useful operations, like z-score standardization, can be performed, at the user's choice, using OpenCV or NumPY.
3. To train the neural network, one of the following files can be used:
   1. **Train.py**: allows you to train the neural network using only the k-Fold.
   2. **TrainWithDataAugmentation.py**: allows you to train the neural network using k-Fold and data augmentation.
   3. The neural networks obtained will be saved in the "Models" folder, while the results will be saved in the "Data" folder.
4. **Test.py**: contains the code to test the best neural network obtained previously (N.B., the code of the best fold must be indicated in the code).
5. **ConfusionMatrix.py**: contains the code necessary to draw the confusion matrix starting from the data obtained in the previous steps.

I also recommend to take a look at the program I created for my thesis, entitled [Facial Expressions Databases Classifier](https://github.com/AntonioMarceddu/Facial_Expressions_Databases_Classifier), as it allows you to easily preprocess the images of some of the most common databases, such as CK+, FER2013 and many others.

## Italian

Codice per addestrare una rete neurale in grado di riconoscere le espressioni facciali con Keras, totalmente commentato in inglese. È stato scritto per la mia tesi di Laurea Magistrale. Esso deve essere eseguito in questo ordine:

1. **CommonVariables.py**: contiene variabili comuni tra i diversi file. Deve essere modificato in modo tale per specificare le cartelle di train e test.
2. **Preprocessing.py**: consente di preprocessare le immagini in input. Alcune operazioni utili, come la standardizzazione z-score, possono essere eseguite, a scelta dell'utente, utilizzando OpenCV o NumPY.
3. Per l'addestramento della rete neurale, si può far ricorso ad uno dei seguenti file:
   1. **Train.py**: consente di addestrare la rete neurale usando solo il k-Fold. 
   2. **TrainWithDataAugmentation.py**: consente di addestrare la rete neurale usando il k-Fold e la data augmentation.
   3. Le reti neurali ottenute verranno salvate nella cartella "Models", mentre i risultati verranno salvati nella cartella "Data".
4. **Test.py**: contiene il codice per testare la miglior rete neurale ottenuta in precedenza (N.B. occorre indicare, nel codice, il numero del miglior fold).
5. **ConfusionMatrix.py**: contiene il codice necessario per disegnare la matrice di confusione partendo dai dati ottenuti nei passaggi precedenti.

Consiglio di dare un'occhiata anche al programma che ho realizzato per la mia tesi, dal titolo [Facial Expressions Databases Classifier](https://github.com/AntonioMarceddu/Facial_Expressions_Databases_Classifier), in quanto consente di preprocessare facilmente le immagini di alcuni dei database più comuni, quali CK+, FER2013 e molti altri.
