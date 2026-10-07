basandosi su https://github.com/baidu/Unlimited-OCR
genera un applicazione desktop gui in python statically linked  che sia funzionante sia per MacOS che per Windows che contiene 2 TAB
il primo tab permette di scegliere un file di tipo PDF  e ne permette un preview  dopo di che usa Unlimited-OCR (https://github.com/baidu/Unlimited-OCR) per convertirlo in markdown o in html
da visualizzare nel tab 2  e consente di esportarlo nei 2 formati .md o .html durante la conversione scegli la densità di conversione da pdf a immagine da un piccolo impostazione di cfg
in partenza se non è già presente fai scaricare il modello nella current dir (mostrando lo stato di avanzamento del download)

LA parte mac deve essere compatibile con i processori M4 , e la parte pc ove possibile deve poter usare cuda

il progetto si chiama AiOCR 


 
realizza il progetto sul mio github con tutto in in inglese su https://github.com/vagrillo/AiOCR
