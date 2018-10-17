#Aufgabe 2:
#Schreibe ein Python Programm, das
#- Die Konvertierung von Temperaturangaben in Celsius nach Fahrenheit oder Kelvin ermöglicht
#- Zuerst wird beim Benutzer abgefragt welche Konvertierung er machen möchte
#- Danach muss der Benutzer eine Temperatur in Celsius angeben
#- Es wir die Temperatur in Fahrenheit oder Kelvin ausgegeben
#Hinweise:
#- Celsius = 5/9 * (Fahrenheit - 32).
#- Celsius = Kelvin - 273.15.
#- Die tiefste mögliche Temperatur ist der absolute Nullpunkt von -273.15 Grad Celsius

def converter():
    print("Geben Sie die Zahl '1' für die Konvertierung in Celcius nach Fahrenheit ein.\n")
    print("Geben Sie die Zahl '2' für die Konvertierung in Celcius nach Kelvin ein.")

    x = input("Eingabe Zahl -> ")
        
    if x == "1":
        print("Celsius -> Fahrenheit\n")
        celsius = input("Geben Sie eine Temperatur in Celsius ein -> ")
        celsius = float(celsius)
        while celsius < -273.15:
            print("Die tiefste mögliche Temperatur ist -273.15 Grad Celsius")
            celsius = input("Bitte geben Sie eine Temperatur in Celsius >= -273.15 Grad ein -> ")
            celsius = float(celsius)

        fahrenheit = celsius * 9/5 + 32
        print(fahrenheit)

    elif x == "2":
        print("Celsius -> Kelvin\n")
        celsius = input("Geben Sie eine Temperatur in Celsius ein -> ")
        celsius = float(celsius)
        while celsius < -273.15:
            print("Die tiefste mögliche Temperatur ist -273.15 Grad Celsius")
            celsius = input("Bitte geben Sie eine Temperatur in Celsius >= -273.15 Grad ein -> ")
            celsius = float(celsius)
            
        kelvin = celsius + 273.15
        print(kelvin)
    else:
        converter()
    
converter()