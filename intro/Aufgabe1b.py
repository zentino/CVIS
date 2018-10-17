#Aufgabe 1:
#Schreibe ein Python Programm, das
#- Bei dem Benutzer eine Zahl n abfragt
#- Den Buchstaben ‚d‘ danach n-Mal ausgibt
#- Der Buchstabe ‚d‘ soll abwechselnd klein und groß geschrieben werden
#1a: verwende dazu eine for-schleife
#1b: verwende dazu eine while-schleife

print("Aufgabe 1b\n")
x = input("Bitte geben Sie eine Zahl n ein > \n")
c = "d"
i = 0
while i < int(x):
    print(c)
    if c.isupper():
        c = c.lower()
    else:
        c = c.upper()
    i = i + 1